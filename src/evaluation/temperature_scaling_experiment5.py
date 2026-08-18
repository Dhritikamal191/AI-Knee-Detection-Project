import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score

from src.models.ordinal_resnet50 import OrdinalResNet50


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5
IMAGE_SIZE = 224
BATCH_SIZE = 32

BASE_PATH = Path(
    "data/raw/KneeXrayMini"
)

VAL_DIR = (
    BASE_PATH /
    "val"
)

TEST_DIR = (
    BASE_PATH /
    "test"
)

MODEL_PATH = Path(
    "artifacts/checkpoints/"
    "best_model_experiment5.pt"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/"
    "calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "=" * 70
)

print(
    "EXPERIMENT 5 TEMPERATURE SCALING"
)

print(
    "=" * 70
)

print(
    f"Using device: {DEVICE}"
)

if DEVICE.type == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        256
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print(
    "\nLoading validation dataset..."
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)

print(
    f"Classes: "
    f"{val_dataset.classes}"
)


print(
    "\nLoading test dataset..."
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)

print(
    f"Classes: "
    f"{test_dataset.classes}"
)


expected_classes = [
    "0",
    "1",
    "2",
    "3",
    "4"
]

if (
    val_dataset.classes
    != expected_classes
):

    raise ValueError(
        "Unexpected validation "
        "class order."
    )

if (
    test_dataset.classes
    != expected_classes
):

    raise ValueError(
        "Unexpected test "
        "class order."
    )


# ============================================================
# DATA LOADERS
# ============================================================

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=(
        DEVICE.type == "cuda"
    )
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=(
        DEVICE.type == "cuda"
    )
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading Experiment 5 model..."
)

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{MODEL_PATH}"
    )


checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE,

    weights_only=False
)

print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'N/A')}"
)


model = OrdinalResNet50(

    num_classes=NUM_CLASSES,

    pretrained=False

).to(DEVICE)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print(
    "Experiment 5 model loaded successfully."
)


# ============================================================
# COLLECT LOGITS
# ============================================================

def collect_logits(
    model,
    loader
):

    all_logits = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                images
            )

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    logits = torch.cat(
        all_logits,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    return (
        logits,
        labels
    )


# ============================================================
# ORDINAL DECODING
# ============================================================

def ordinal_to_grade(
    logits
):

    probabilities = torch.sigmoid(
        logits
    )

    grades = (
        probabilities >= 0.5
    ).sum(
        dim=1
    )

    return grades.long()


# ============================================================
# ORDINAL → CLASS PROBABILITIES
# ============================================================

def ordinal_to_class_probabilities(
    logits
):

    threshold_probabilities = (
        torch.sigmoid(
            logits
        )
    )

    p_ge_1 = (
        threshold_probabilities[:, 0]
    )

    p_ge_2 = (
        threshold_probabilities[:, 1]
    )

    p_ge_3 = (
        threshold_probabilities[:, 2]
    )

    p_ge_4 = (
        threshold_probabilities[:, 3]
    )

    class_probabilities = torch.stack(

        [

            1.0 - p_ge_1,

            p_ge_1 - p_ge_2,

            p_ge_2 - p_ge_3,

            p_ge_3 - p_ge_4,

            p_ge_4

        ],

        dim=1
    )

    class_probabilities = torch.clamp(
        class_probabilities,
        min=0.0
    )

    class_probabilities = (

        class_probabilities /

        class_probabilities.sum(
            dim=1,
            keepdim=True
        ).clamp(
            min=1e-8
        )

    )

    return class_probabilities


# ============================================================
# LOAD VALIDATION LOGITS
# ============================================================

print(
    "\nGenerating validation logits..."
)

val_logits, val_labels = collect_logits(
    model,
    val_loader
)

print(
    f"Validation predictions: "
    f"{len(val_labels)}"
)


# ============================================================
# LOAD TEST LOGITS
# ============================================================

print(
    "\nGenerating test logits..."
)

test_logits, test_labels = collect_logits(
    model,
    test_loader
)

print(
    f"Test predictions: "
    f"{len(test_labels)}"
)


# ============================================================
# TEMPERATURE PARAMETER
# ============================================================

class TemperatureScaler(
    nn.Module
):

    def __init__(
        self,
        initial_temperature=1.0
    ):

        super().__init__()

        self.temperature = nn.Parameter(

            torch.tensor(

                [
                    float(
                        initial_temperature
                    )
                ],

                dtype=torch.float32
            )
        )

    def forward(
        self,
        logits
    ):

        return (
            logits /
            self.temperature.clamp(
                min=0.05
            )
        )


# ============================================================
# VALIDATION TARGETS
# ============================================================

val_logits = val_logits.to(
    DEVICE
)

val_labels = val_labels.to(
    DEVICE
)


# ============================================================
# FIT TEMPERATURE
# ============================================================

print(
    "\nFitting temperature "
    "on validation set..."
)


temperature_scaler = TemperatureScaler(
    initial_temperature=1.0
).to(DEVICE)


optimizer = torch.optim.LBFGS(

    [
        temperature_scaler.temperature
    ],

    lr=0.01,

    max_iter=100,

    line_search_fn="strong_wolfe"
)


def closure():

    optimizer.zero_grad()

    scaled_logits = (
        temperature_scaler(
            val_logits
        )
    )

    threshold_probabilities = (
        torch.sigmoid(
            scaled_logits
        )
    )

    ordinal_targets = torch.stack(

        [

            (val_labels >= 1).float(),

            (val_labels >= 2).float(),

            (val_labels >= 3).float(),

            (val_labels >= 4).float()

        ],

        dim=1
    )

    loss = nn.functional.binary_cross_entropy(
        threshold_probabilities,
        ordinal_targets
    )

    loss.backward()

    return loss


optimizer.step(
    closure
)


temperature = float(
    temperature_scaler.temperature
    .detach()
    .cpu()
    .item()
)

temperature = max(
    temperature,
    0.05
)


print(
    f"\nLearned temperature: "
    f"{temperature:.6f}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_calibration(
    logits,
    labels,
    temperature
):

    logits = logits.to(
        DEVICE
    )

    labels = labels.to(
        DEVICE
    )

    scaled_logits = (
        logits /
        temperature
    )

    class_probabilities = (
        ordinal_to_class_probabilities(
            scaled_logits
        )
    )

    predictions = (
        ordinal_to_grade(
            scaled_logits
        )
    )

    confidences = torch.max(
        class_probabilities,
        dim=1
    ).values

    probabilities_np = (
        class_probabilities
        .detach()
        .cpu()
        .numpy()
    )

    predictions_np = (
        predictions
        .detach()
        .cpu()
        .numpy()
    )

    labels_np = (
        labels
        .detach()
        .cpu()
        .numpy()
    )

    confidences_np = (
        confidences
        .detach()
        .cpu()
        .numpy()
    )

    accuracy = accuracy_score(
        labels_np,
        predictions_np
    )

    correct = (
        predictions_np
        == labels_np
    )

    # --------------------------------------------------------
    # ECE
    # --------------------------------------------------------

    ece = 0.0
    mce = 0.0

    bins = []

    for bin_index in range(
        10
    ):

        lower = (
            bin_index / 10.0
        )

        upper = (
            (bin_index + 1) / 10.0
        )

        if bin_index == 9:

            mask = (
                (confidences_np >= lower)
                &
                (confidences_np <= upper)
            )

        else:

            mask = (
                (confidences_np >= lower)
                &
                (confidences_np < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        bin_confidence = float(
            confidences_np[mask].mean()
        )

        bin_accuracy = float(
            correct[mask].mean()
        )

        gap = abs(
            bin_confidence
            -
            bin_accuracy
        )

        ece += (
            count /
            len(labels_np)
        ) * gap

        mce = max(
            mce,
            gap
        )

        bins.append({

            "bin":
                bin_index + 1,

            "samples":
                count,

            "confidence":
                bin_confidence,

            "accuracy":
                bin_accuracy,

            "gap":
                gap
        })

    # --------------------------------------------------------
    # Brier score
    # --------------------------------------------------------

    one_hot = np.zeros_like(
        probabilities_np
    )

    one_hot[
        np.arange(
            len(labels_np)
        ),
        labels_np
    ] = 1.0

    brier_score = float(
        np.mean(
            np.sum(
                (
                    probabilities_np
                    -
                    one_hot
                ) ** 2,
                axis=1
            )
        )
    )

    # --------------------------------------------------------
    # Confidence statistics
    # --------------------------------------------------------

    if correct.any():

        mean_conf_correct = float(
            confidences_np[
                correct
            ].mean()
        )

    else:

        mean_conf_correct = 0.0

    if (~correct).any():

        mean_conf_incorrect = float(
            confidences_np[
                ~correct
            ].mean()
        )

    else:

        mean_conf_incorrect = 0.0

    high_confidence = (
        confidences_np >= 0.80
    )

    high_conf_count = int(
        high_confidence.sum()
    )

    if high_conf_count > 0:

        high_conf_accuracy = float(
            correct[
                high_confidence
            ].mean()
        )

    else:

        high_conf_accuracy = 0.0

    return {

        "accuracy":
            float(accuracy),

        "ece":
            float(ece),

        "mce":
            float(mce),

        "brier_score":
            brier_score,

        "mean_confidence_correct":
            mean_conf_correct,

        "mean_confidence_incorrect":
            mean_conf_incorrect,

        "high_confidence_samples":
            high_conf_count,

        "high_confidence_accuracy":
            high_conf_accuracy,

        "high_confidence_error_rate":
            (
                1.0 -
                high_conf_accuracy
            ),

        "bins":
            bins,

        "predictions":
            predictions_np,

        "probabilities":
            probabilities_np,

        "confidences":
            confidences_np
    }


# ============================================================
# BEFORE CALIBRATION
# ============================================================

print(
    "\nEvaluating BEFORE calibration..."
)

before = evaluate_calibration(

    test_logits,

    test_labels,

    temperature=1.0
)


# ============================================================
# AFTER CALIBRATION
# ============================================================

print(
    "Evaluating AFTER calibration..."
)

after = evaluate_calibration(

    test_logits,

    test_labels,

    temperature=temperature
)


# ============================================================
# RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "EXPERIMENT 5 TEMPERATURE SCALING RESULTS"
)

print(
    "=" * 70
)

print(
    f"\nLearned Temperature: "
    f"{temperature:.6f}"
)


print(
    "\nMetric"
    f"{'Before':>18}"
    f"{'After':>18}"
    f"{'Change':>18}"
)

print(
    "-" * 70
)


def print_metric(
    name,
    before_value,
    after_value
):

    change = (
        after_value
        -
        before_value
    )

    print(
        f"{name:<28}"
        f"{before_value:>12.4f}"
        f"{after_value:>18.4f}"
        f"{change:>18.4f}"
    )


print_metric(
    "Accuracy",
    before["accuracy"],
    after["accuracy"]
)

print_metric(
    "ECE",
    before["ece"],
    after["ece"]
)

print_metric(
    "MCE",
    before["mce"],
    after["mce"]
)

print_metric(
    "Brier Score",
    before["brier_score"],
    after["brier_score"]
)

print_metric(
    "Mean Confidence Correct",
    before[
        "mean_confidence_correct"
    ],
    after[
        "mean_confidence_correct"
    ]
)

print_metric(
    "Mean Confidence Incorrect",
    before[
        "mean_confidence_incorrect"
    ],
    after[
        "mean_confidence_incorrect"
    ]
)

print_metric(
    "High Confidence Accuracy",
    before[
        "high_confidence_accuracy"
    ],
    after[
        "high_confidence_accuracy"
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "Experiment 5",

    "temperature":
        temperature,

    "test_samples":
        len(test_labels),

    "before_calibration": {

        key: value

        for key, value
        in before.items()

        if key not in [
            "predictions",
            "probabilities",
            "confidences"
        ]
    },

    "after_calibration": {

        key: value

        for key, value
        in after.items()

        if key not in [
            "predictions",
            "probabilities",
            "confidences"
        ]
    }
}


json_path = (
    OUTPUT_DIR /
    "experiment5_temperature_scaling.json"
)


with open(
    json_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_rows = []


for i in range(
    len(test_labels)
):

    row = {

        "index":
            i,

        "true_grade":
            int(
                test_labels[i]
            ),

        "predicted_grade_before":
            int(
                before[
                    "predictions"
                ][i]
            ),

        "predicted_grade_after":
            int(
                after[
                    "predictions"
                ][i]
            ),

        "correct_before":
            bool(
                before[
                    "predictions"
                ][i]
                ==
                int(
                    test_labels[i]
                )
            ),

        "correct_after":
            bool(
                after[
                    "predictions"
                ][i]
                ==
                int(
                    test_labels[i]
                )
            ),

        "confidence_before":
            float(
                before[
                    "confidences"
                ][i]
            ),

        "confidence_after":
            float(
                after[
                    "confidences"
                ][i]
            )
    }

    for grade in range(
        NUM_CLASSES
    ):

        row[
            f"prob_before_grade_{grade}"
        ] = float(
            before[
                "probabilities"
            ][i][grade]
        )

        row[
            f"prob_after_grade_{grade}"
        ] = float(
            after[
                "probabilities"
            ][i][grade]
        )

    prediction_rows.append(
        row
    )


predictions_df = pd.DataFrame(
    prediction_rows
)


csv_path = (
    OUTPUT_DIR /
    "experiment5_temperature_predictions.csv"
)


predictions_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TEMPERATURE SCALING COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nTemperature:"
)

print(
    temperature
)

print(
    "\nResults JSON:"
)

print(
    json_path
)

print(
    "\nPrediction CSV:"
)

print(
    csv_path
)

print(
    "\n"
    + "=" * 70
)