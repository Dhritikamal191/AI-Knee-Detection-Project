# ============================================================
# EXPERIMENT 5
# CALIBRATION ANALYSIS
#
# Ordinal ResNet50
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.training.train_experiment5 import OrdinalResNet50


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 5

IMAGE_SIZE = 224

BATCH_SIZE = 32

MODEL_PATH = Path(
    "artifacts/checkpoints/best_model_experiment5.pt"
)

TEST_DIR = Path(
    "data/raw/KneeXrayMini/test"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)

print(
    "EXPERIMENT 5 CALIBRATION ANALYSIS"
)

print("=" * 70)

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
# LOAD TEST DATASET
# ============================================================

print(
    "\nLoading test dataset..."
)

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"Test directory not found:\n"
        f"{TEST_DIR}"
    )


test_dataset = ImageFolder(
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


# ============================================================
# CHECK CLASS ORDER
# ============================================================

expected_classes = [
    "0",
    "1",
    "2",
    "3",
    "4"
]

if test_dataset.classes != expected_classes:

    raise ValueError(
        "\nUnexpected class order.\n"
        f"Expected: {expected_classes}\n"
        f"Found: {test_dataset.classes}"
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


# ------------------------------------------------------------
# PyTorch 2.6+
#
# The checkpoint contains objects such as NumPy scalars,
# therefore weights_only=False is required for this trusted
# locally-created checkpoint.
# ------------------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'N/A')}"
)


# ============================================================
# CREATE ORDINAL MODEL
# ============================================================

model = OrdinalResNet50(
    num_classes=NUM_CLASSES,
    pretrained=False
).to(DEVICE)


# ============================================================
# LOAD STATE DICT
# ============================================================

if "model_state_dict" not in checkpoint:

    raise KeyError(
        "model_state_dict not found "
        "inside Experiment 5 checkpoint."
    )


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
# ORDINAL → CLASS PROBABILITIES
# ============================================================

def ordinal_to_class_probabilities(
    outputs
):
    """
    Convert four ordinal threshold probabilities into
    five mutually exclusive grade probabilities.

    outputs:

        output[0] -> Grade >= 1
        output[1] -> Grade >= 2
        output[2] -> Grade >= 3
        output[3] -> Grade >= 4

    Class probabilities:

        Grade 0 = 1 - P(>=1)
        Grade 1 = P(>=1) - P(>=2)
        Grade 2 = P(>=2) - P(>=3)
        Grade 3 = P(>=3) - P(>=4)
        Grade 4 = P(>=4)
    """

    threshold_probabilities = (
        torch.sigmoid(
            outputs
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

    # --------------------------------------------------------
    # Numerical protection
    # --------------------------------------------------------

    class_probabilities = torch.clamp(
        class_probabilities,
        min=0.0
    )

    # --------------------------------------------------------
    # Normalize so probabilities sum to 1
    # --------------------------------------------------------

    class_probabilities = (
        class_probabilities /
        class_probabilities.sum(
            dim=1,
            keepdim=True
        ).clamp(
            min=1e-8
        )
    )

    return (
        threshold_probabilities,
        class_probabilities
    )


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)

all_true = []

all_predictions = []

all_confidences = []

all_probabilities = []

all_threshold_probabilities = []

all_paths = []


with torch.no_grad():

    for (
        images,
        labels
    ) in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            images
        )

        (
            threshold_probabilities,
            class_probabilities
        ) = ordinal_to_class_probabilities(
            outputs
        )

        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        # IMPORTANT:
        # Experiment 5 uses the exact same ordinal decoding
        # rule as train_experiment5.py:
        #
        # sigmoid(logits) >= 0.5
        # Number of passed thresholds = predicted grade

        predictions = (
        threshold_probabilities >= 0.5
        ).sum(
        dim=1
        ).long()


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        # Build confidence using the probability of the
        # predicted ordinal grade.

        confidences = torch.zeros(
        predictions.shape,
        device=predictions.device,
        dtype=torch.float32
        )

        # Grade 0:
        # Confidence = P(Grade < 1)
        mask = predictions == 0

        confidences[mask] = (
        1.0 -
        threshold_probabilities[
        mask, 0
        ]
        )


        # Grade 1:
        # Confidence = min(
        #     P(Grade >= 1),
        #     P(Grade < 2)
        # )

        mask = predictions == 1

        confidences[mask] = torch.minimum(
        threshold_probabilities[
        mask, 0
        ],
        1.0 -
        threshold_probabilities[
        mask, 1
        ]
        )


        # Grade 2:
        # Confidence = min(
        #     P(Grade >= 2),
        #     P(Grade < 3)
        # )

        mask = predictions == 2

        confidences[mask] = torch.minimum(
        threshold_probabilities[
        mask, 1
        ],
        1.0 -
        threshold_probabilities[
        mask, 2
        ]
        )


        # Grade 3:
        # Confidence = min(
        #     P(Grade >= 3),
        #     P(Grade < 4)
        # )

        mask = predictions == 3

        confidences[mask] = torch.minimum(
        threshold_probabilities[
        mask, 2
        ],
        1.0 -
        threshold_probabilities[
        mask, 3
        ]
        )


        # Grade 4:
        # Confidence = P(Grade >= 4)

        mask = predictions == 4

        confidences[mask] = (
        threshold_probabilities[
        mask, 3
        ]
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_true.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_confidences.extend(
            confidences.cpu().numpy()
        )

        all_probabilities.extend(
            class_probabilities.cpu().numpy()
        )

        all_threshold_probabilities.extend(
            threshold_probabilities.cpu().numpy()
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(
    all_true,
    dtype=int
)

y_pred = np.array(
    all_predictions,
    dtype=int
)

confidences = np.array(
    all_confidences,
    dtype=float
)

probabilities = np.array(
    all_probabilities,
    dtype=float
)

threshold_probabilities = np.array(
    all_threshold_probabilities,
    dtype=float
)


# ============================================================
# BASIC CHECK
# ============================================================

print(
    f"\nPredictions generated: "
    f"{len(y_true)}"
)

if len(y_true) != len(test_dataset):

    raise RuntimeError(
        "Number of predictions does not "
        "match number of test images."
    )


# ============================================================
# ACCURACY
# ============================================================

accuracy = np.mean(
    y_true == y_pred
)


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def calculate_ece(
    true_labels,
    predictions,
    confidence,
    num_bins=10
):

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    ece = 0.0

    bin_results = []

    total = len(
        true_labels
    )

    for i in range(
        num_bins
    ):

        lower = bin_edges[i]

        upper = bin_edges[i + 1]

        if i == num_bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        count = np.sum(
            mask
        )

        if count == 0:

            bin_results.append({

                "bin":
                    i + 1,

                "lower":
                    float(lower),

                "upper":
                    float(upper),

                "count":
                    0,

                "accuracy":
                    None,

                "confidence":
                    None,

                "gap":
                    None
            })

            continue

        bin_accuracy = np.mean(
            true_labels[mask]
            ==
            predictions[mask]
        )

        bin_confidence = np.mean(
            confidence[mask]
        )

        gap = abs(
            bin_accuracy -
            bin_confidence
        )

        ece += (
            count /
            total
        ) * gap

        bin_results.append({

            "bin":
                i + 1,

            "lower":
                float(lower),

            "upper":
                float(upper),

            "count":
                int(count),

            "accuracy":
                float(
                    bin_accuracy
                ),

            "confidence":
                float(
                    bin_confidence
                ),

            "gap":
                float(gap)
        })

    return (
        float(ece),
        bin_results
    )


# ============================================================
# ECE
# ============================================================

ece, calibration_bins = (
    calculate_ece(
        y_true,
        y_pred,
        confidences,
        num_bins=10
    )
)


# ============================================================
# MAXIMUM CALIBRATION ERROR
# ============================================================

valid_gaps = [

    item["gap"]

    for item in calibration_bins

    if item["gap"] is not None
]


if valid_gaps:

    mce = max(
        valid_gaps
    )

else:

    mce = 0.0


# ============================================================
# MULTICLASS BRIER SCORE
# ============================================================

one_hot = np.zeros_like(
    probabilities
)

one_hot[
    np.arange(
        len(y_true)
    ),
    y_true
] = 1.0


brier_score = np.mean(
    np.sum(
        (
            probabilities -
            one_hot
        ) ** 2,
        axis=1
    )
)


# ============================================================
# CONFIDENCE STATISTICS
# ============================================================

correct_mask = (
    y_true == y_pred
)

incorrect_mask = (
    ~correct_mask
)


if np.any(correct_mask):

    mean_confidence_correct = (
        np.mean(
            confidences[
                correct_mask
            ]
        )
    )

else:

    mean_confidence_correct = 0.0


if np.any(incorrect_mask):

    mean_confidence_incorrect = (
        np.mean(
            confidences[
                incorrect_mask
            ]
        )
    )

else:

    mean_confidence_incorrect = 0.0


# ============================================================
# HIGH-CONFIDENCE ERROR RATE
# ============================================================

high_confidence_mask = (
    confidences >= 0.80
)

high_confidence_count = np.sum(
    high_confidence_mask
)

if high_confidence_count > 0:

    high_confidence_accuracy = (
        np.mean(
            correct_mask[
                high_confidence_mask
            ]
        )
    )

    high_confidence_error_rate = (
        1.0 -
        high_confidence_accuracy
    )

else:

    high_confidence_accuracy = 0.0

    high_confidence_error_rate = 0.0


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_rows = []


for index in range(
    len(y_true)
):

    prediction_rows.append({

        "index":
            int(index),

        "true_grade":
            int(y_true[index]),

        "predicted_grade":
            int(y_pred[index]),

        "correct":
            bool(
                y_true[index]
                ==
                y_pred[index]
            ),

        "confidence":
            float(
                confidences[index]
            ),

        "prob_grade_0":
            float(
                probabilities[
                    index,
                    0
                ]
            ),

        "prob_grade_1":
            float(
                probabilities[
                    index,
                    1
                ]
            ),

        "prob_grade_2":
            float(
                probabilities[
                    index,
                    2
                ]
            ),

        "prob_grade_3":
            float(
                probabilities[
                    index,
                    3
                ]
            ),

        "prob_grade_4":
            float(
                probabilities[
                    index,
                    4
                ]
            ),

        "prob_grade_ge_1":
            float(
                threshold_probabilities[
                    index,
                    0
                ]
            ),

        "prob_grade_ge_2":
            float(
                threshold_probabilities[
                    index,
                    1
                ]
            ),

        "prob_grade_ge_3":
            float(
                threshold_probabilities[
                    index,
                    2
                ]
            ),

        "prob_grade_ge_4":
            float(
                threshold_probabilities[
                    index,
                    3
                ]
            )
    })


predictions_df = pd.DataFrame(
    prediction_rows
)


predictions_csv = (
    OUTPUT_DIR /
    "experiment5_test_predictions.csv"
)


predictions_df.to_csv(
    predictions_csv,
    index=False
)


# ============================================================
# CALIBRATION JSON
# ============================================================

calibration_results = {

    "experiment":
        "Experiment 5",

    "model":
        "Ordinal ResNet50",

    "test_samples":
        int(len(y_true)),

    "accuracy":
        float(accuracy),

    "expected_calibration_error":
        float(ece),

    "maximum_calibration_error":
        float(mce),

    "brier_score":
        float(brier_score),

    "mean_confidence_correct":
        float(
            mean_confidence_correct
        ),

    "mean_confidence_incorrect":
        float(
            mean_confidence_incorrect
        ),

    "high_confidence_threshold":
        0.80,

    "high_confidence_samples":
        int(
            high_confidence_count
        ),

    "high_confidence_accuracy":
        float(
            high_confidence_accuracy
        ),

    "high_confidence_error_rate":
        float(
            high_confidence_error_rate
        ),

    "calibration_bins":
        calibration_bins
}


json_path = (
    OUTPUT_DIR /
    "experiment5_calibration.json"
)


with open(
    json_path,
    "w"
) as f:

    json.dump(
        calibration_results,
        f,
        indent=4
    )


# ============================================================
# RELIABILITY DIAGRAM
# ============================================================

bin_confidences = []

bin_accuracies = []

bin_counts = []


for item in calibration_bins:

    if item["confidence"] is not None:

        bin_confidences.append(
            item["confidence"]
        )

        bin_accuracies.append(
            item["accuracy"]
        )

        bin_counts.append(
            item["count"]
        )


plt.figure(
    figsize=(8, 7)
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)


if bin_confidences:

    plt.plot(
        bin_confidences,
        bin_accuracies,
        marker="o",
        label="Experiment 5"
    )


plt.xlabel(
    "Mean Predicted Confidence"
)

plt.ylabel(
    "Observed Accuracy"
)

plt.title(
    "Experiment 5 Reliability Diagram"
)

plt.xlim(
    0,
    1
)

plt.ylim(
    0,
    1
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()


reliability_path = (
    OUTPUT_DIR /
    "calibration_curve_experiment5.png"
)


plt.savefig(
    reliability_path,
    dpi=300
)

plt.close()


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.hist(
    confidences,
    bins=10
)


plt.xlabel(
    "Prediction Confidence"
)

plt.ylabel(
    "Number of Test Images"
)

plt.title(
    "Experiment 5 Confidence Distribution"
)

plt.xlim(
    0,
    1
)

plt.tight_layout()


confidence_path = (
    OUTPUT_DIR /
    "confidence_distribution_experiment5.png"
)


plt.savefig(
    confidence_path,
    dpi=300
)

plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")

print("=" * 70)

print(
    "EXPERIMENT 5 CALIBRATION RESULTS"
)

print("=" * 70)

print(
    f"Test Samples: "
    f"{len(y_true)}"
)

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Expected Calibration Error (ECE): "
    f"{ece:.4f}"
)

print(
    f"Maximum Calibration Error (MCE): "
    f"{mce:.4f}"
)

print(
    f"Brier Score: "
    f"{brier_score:.4f}"
)

print(
    f"Mean Confidence - Correct: "
    f"{mean_confidence_correct:.4f}"
)

print(
    f"Mean Confidence - Incorrect: "
    f"{mean_confidence_incorrect:.4f}"
)

print(
    f"High Confidence Samples (>=80%): "
    f"{high_confidence_count}"
)

print(
    f"High Confidence Accuracy: "
    f"{high_confidence_accuracy:.4f}"
)

print(
    f"High Confidence Error Rate: "
    f"{high_confidence_error_rate:.4f}"
)


# ============================================================
# CALIBRATION TABLE
# ============================================================

print("\n")

print(
    "CALIBRATION BINS"
)

print("-" * 70)

print(
    f"{'Bin':<8}"
    f"{'Samples':<10}"
    f"{'Confidence':<15}"
    f"{'Accuracy':<15}"
    f"{'Gap':<10}"
)

print("-" * 70)


for item in calibration_bins:

    if item["confidence"] is None:

        continue

    print(

        f"{item['bin']:<8}"

        f"{item['count']:<10}"

        f"{item['confidence']:<15.4f}"

        f"{item['accuracy']:<15.4f}"

        f"{item['gap']:<10.4f}"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n")

print("=" * 70)

print(
    "CALIBRATION ANALYSIS COMPLETE"
)

print("=" * 70)

print(
    "\nPrediction CSV:"
)

print(
    predictions_csv
)

print(
    "\nCalibration JSON:"
)

print(
    json_path
)

print(
    "\nReliability diagram:"
)

print(
    reliability_path
)

print(
    "\nConfidence distribution:"
)

print(
    confidence_path
)

print(
    "\n"
)

print("=" * 70)