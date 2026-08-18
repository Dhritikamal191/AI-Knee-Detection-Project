from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    mean_absolute_error
)

import matplotlib.pyplot as plt

from src.models.ordinal_resnet50 import OrdinalResNet50
from src.models.training_config import get_device, BATCH_SIZE


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]

IMAGE_SIZE = 224

BASE_PATH = Path(
    "data/raw/KneeXrayMini"
)

MODEL_PATH = Path(
    "artifacts/checkpoints/"
    "best_model_experiment5.pt"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([

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
# ORDINAL CLASS PROBABILITIES
# ============================================================

def ordinal_to_class_probabilities(
    logits
):

    threshold_probabilities = torch.sigmoid(
        logits
    )

    p_ge_1 = threshold_probabilities[:, 0]
    p_ge_2 = threshold_probabilities[:, 1]
    p_ge_3 = threshold_probabilities[:, 2]
    p_ge_4 = threshold_probabilities[:, 3]

    probabilities = torch.stack(

        [
            1.0 - p_ge_1,

            p_ge_1 - p_ge_2,

            p_ge_2 - p_ge_3,

            p_ge_3 - p_ge_4,

            p_ge_4
        ],

        dim=1
    )

    probabilities = torch.clamp(
        probabilities,
        min=0.0,
        max=1.0
    )

    probabilities = (
        probabilities /
        probabilities.sum(
            dim=1,
            keepdim=True
        ).clamp(
            min=1e-8
        )
    )

    return probabilities


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    device = get_device()

    print(
        f"Using device: {device}"
    )

    model = OrdinalResNet50(
        num_classes=NUM_CLASSES,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(

        MODEL_PATH,

        map_location=device,

        weights_only=False
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model, device


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "EXPERIMENT 5 ERROR ANALYSIS"
)

print(
    "=" * 70
)

print(
    "\nLoading test dataset..."
)

test_dataset = datasets.ImageFolder(

    BASE_PATH / "test",

    transform=test_transform
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)

print(
    f"Classes: "
    f"{test_dataset.classes}"
)


if test_dataset.classes != CLASS_NAMES:

    raise ValueError(

        "\nUnexpected class order.\n"

        f"Expected: {CLASS_NAMES}\n"

        f"Found: {test_dataset.classes}"
    )


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

model, device = load_model()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print(
    "\nGenerating test predictions..."
)

all_labels = []

all_predictions = []

all_confidences = []

all_probabilities = []

all_paths = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        logits = model(
            images
        )

        predictions = ordinal_to_grade(
            logits
        )

        class_probabilities = (
            ordinal_to_class_probabilities(
                logits
            )
        )

        confidence, _ = (
            class_probabilities.max(
                dim=1
            )
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_confidences.extend(
            confidence.cpu().numpy()
        )

        all_probabilities.extend(
            class_probabilities.cpu().numpy()
        )


# ============================================================
# CONVERT ARRAYS
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_confidences = np.array(
    all_confidences
)

all_probabilities = np.array(
    all_probabilities
)

all_paths = [
    path
    for path, _ in test_dataset.samples
]


print(
    f"Predictions generated: "
    f"{len(all_predictions)}"
)


# ============================================================
# BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(

    all_labels,

    all_predictions,

    average="macro",

    zero_division=0
)

balanced_acc = balanced_accuracy_score(

    all_labels,

    all_predictions
)

qwk = cohen_kappa_score(

    all_labels,

    all_predictions,

    weights="quadratic"
)

mae = mean_absolute_error(

    all_labels,

    all_predictions
)


print(
    "\n"
    + "=" * 70
)

print(
    "OVERALL METRICS"
)

print(
    "=" * 70
)

print(
    f"Accuracy:             {accuracy:.4f}"
)

print(
    f"Macro F1:             {macro_f1:.4f}"
)

print(
    f"Balanced Accuracy:    {balanced_acc:.4f}"
)

print(
    f"Quadratic Kappa:      {qwk:.4f}"
)

print(
    f"Mean Absolute Error:  {mae:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions,

    labels=range(NUM_CLASSES)
)


print(
    "\n"
    + "=" * 70
)

print(
    "CONFUSION MATRIX"
)

print(
    "=" * 70
)

print(
    cm
)


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = (

    cm.astype(float)

    /
    cm.sum(
        axis=1,
        keepdims=True
    ).clip(
        min=1
    )
)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(

    matrix,

    filename,

    title,

    fmt

):

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        matrix
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Predicted Grade"
    )

    ax.set_ylabel(
        "True Grade"
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_xticklabels(
        CLASS_NAMES
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            ax.text(

                j,

                i,

                format(
                    matrix[i, j],
                    fmt
                ),

                ha="center",

                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    fig.savefig(

        OUTPUT_DIR / filename,

        dpi=200
    )

    plt.close(
        fig
    )


plot_confusion_matrix(

    cm,

    "confusion_matrix.png",

    "Experiment 5 Confusion Matrix",

    "d"
)


plot_confusion_matrix(

    cm_normalized,

    "normalized_confusion_matrix.png",

    "Experiment 5 Normalized Confusion Matrix",

    ".2f"
)


# ============================================================
# PER CLASS METRICS
# ============================================================

report = classification_report(

    all_labels,

    all_predictions,

    labels=range(NUM_CLASSES),

    target_names=CLASS_NAMES,

    output_dict=True,

    zero_division=0
)


print(
    "\n"
    + "=" * 70
)

print(
    "PER-CLASS METRICS"
)

print(
    "=" * 70
)


per_class_metrics = {}


for class_name in CLASS_NAMES:

    metrics = report[
        class_name
    ]

    per_class_metrics[
        class_name
    ] = {

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1":
            metrics["f1-score"],

        "support":
            int(
                metrics["support"]
            )
    }

    print(

        f"Grade {class_name} | "

        f"Precision: "
        f"{metrics['precision']:.4f} | "

        f"Recall: "
        f"{metrics['recall']:.4f} | "

        f"F1: "
        f"{metrics['f1-score']:.4f}"
    )


# ============================================================
# MISCLASSIFICATION ANALYSIS
# ============================================================

misclassified_rows = []

error_pairs = {}


for i in range(
    len(all_labels)
):

    true_grade = int(
        all_labels[i]
    )

    predicted_grade = int(
        all_predictions[i]
    )

    if true_grade != predicted_grade:

        pair = (
            f"{true_grade}->{predicted_grade}"
        )

        error_pairs[pair] = (
            error_pairs.get(
                pair,
                0
            ) + 1
        )

        misclassified_rows.append({

            "index":
                i,

            "image":
                all_paths[i],

            "true_grade":
                true_grade,

            "predicted_grade":
                predicted_grade,

            "error_distance":
                abs(
                    true_grade -
                    predicted_grade
                ),

            "confidence":
                float(
                    all_confidences[i]
                ),

            "confidence_percent":
                float(
                    all_confidences[i] * 100
                ),

            "correct":
                False
        })


misclassified_df = pd.DataFrame(
    misclassified_rows
)


misclassified_df = (
    misclassified_df.sort_values(
        "confidence",
        ascending=False
    )
)


misclassified_df.to_csv(

    OUTPUT_DIR /
    "misclassified_predictions.csv",

    index=False
)


# ============================================================
# ERROR PAIRS
# ============================================================

sorted_error_pairs = dict(

    sorted(

        error_pairs.items(),

        key=lambda x: x[1],

        reverse=True
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "MOST COMMON MISCLASSIFICATIONS"
)

print(
    "=" * 70
)


for pair, count in (
    list(
        sorted_error_pairs.items()
    )[:15]
):

    print(
        f"{pair}: {count}"
    )


# ============================================================
# ADJACENT VS SEVERE ERRORS
# ============================================================

error_distances = np.abs(

    all_labels -
    all_predictions
)


incorrect_mask = (
    all_labels !=
    all_predictions
)


adjacent_errors = int(

    np.sum(

        (
            error_distances ==
            1
        )
        &
        incorrect_mask
    )
)


two_grade_errors = int(

    np.sum(

        (
            error_distances ==
            2
        )
        &
        incorrect_mask
    )
)


severe_errors = int(

    np.sum(

        (
            error_distances >=
            3
        )
        &
        incorrect_mask
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "ERROR SEVERITY"
)

print(
    "=" * 70
)

print(
    f"Adjacent-grade errors (1 grade): "
    f"{adjacent_errors}"
)

print(
    f"Two-grade errors: "
    f"{two_grade_errors}"
)

print(
    f"Severe errors (>=3 grades): "
    f"{severe_errors}"
)


# ============================================================
# OVERCONFIDENT ERRORS
# ============================================================

overconfident_mask = (

    incorrect_mask

    &
    (
        all_confidences >=
        0.90
    )
)


overconfident_errors = int(

    np.sum(
        overconfident_mask
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "OVERCONFIDENT ERRORS"
)

print(
    "=" * 70
)

print(
    f"Incorrect predictions with "
    f"confidence >= 90%: "
    f"{overconfident_errors}"
)


# ============================================================
# ERROR CONFIDENCE
# ============================================================

correct_mask = (
    all_labels ==
    all_predictions
)


mean_correct_confidence = float(

    all_confidences[
        correct_mask
    ].mean()
)


mean_incorrect_confidence = float(

    all_confidences[
        incorrect_mask
    ].mean()
)


print(
    f"Mean confidence - correct: "
    f"{mean_correct_confidence:.4f}"
)

print(
    f"Mean confidence - incorrect: "
    f"{mean_incorrect_confidence:.4f}"
)


# ============================================================
# MISCLASSIFICATION DISTRIBUTION
# ============================================================

distance_counts = {

    "1_grade":
        adjacent_errors,

    "2_grades":
        two_grade_errors,

    "3_or_more":
        severe_errors
}


fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.bar(

    list(
        distance_counts.keys()
    ),

    list(
        distance_counts.values()
    )
)


ax.set_title(
    "Experiment 5 Misclassification Distance"
)

ax.set_xlabel(
    "Prediction Error Distance"
)

ax.set_ylabel(
    "Number of Errors"
)


fig.tight_layout()


fig.savefig(

    OUTPUT_DIR /
    "misclassification_distribution.png",

    dpi=200
)


plt.close(
    fig
)


# ============================================================
# SAVE ERROR ANALYSIS JSON
# ============================================================

error_analysis = {

    "experiment":
        "Experiment 5",

    "test_samples":
        int(
            len(all_labels)
        ),

    "accuracy":
        float(
            accuracy
        ),

    "macro_f1":
        float(
            macro_f1
        ),

    "balanced_accuracy":
        float(
            balanced_acc
        ),

    "quadratic_weighted_kappa":
        float(
            qwk
        ),

    "mean_absolute_error":
        float(
            mae
        ),

    "correct_predictions":
        int(
            correct_mask.sum()
        ),

    "incorrect_predictions":
        int(
            incorrect_mask.sum()
        ),

    "mean_confidence_correct":
        mean_correct_confidence,

    "mean_confidence_incorrect":
        mean_incorrect_confidence,

    "overconfident_errors_ge_90":
        overconfident_errors,

    "adjacent_errors":
        adjacent_errors,

    "two_grade_errors":
        two_grade_errors,

    "severe_errors_ge_3":
        severe_errors,

    "error_pairs":
        sorted_error_pairs,

    "per_class_metrics":
        per_class_metrics
}


with open(

    OUTPUT_DIR /
    "error_analysis.json",

    "w"

) as f:

    json.dump(

        error_analysis,

        f,

        indent=4
    )


# ============================================================
# SAVE PER-CLASS JSON
# ============================================================

with open(

    OUTPUT_DIR /
    "per_class_metrics.json",

    "w"

) as f:

    json.dump(

        per_class_metrics,

        f,

        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ERROR ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nGenerated files:"
)

print(
    OUTPUT_DIR /
    "confusion_matrix.png"
)

print(
    OUTPUT_DIR /
    "normalized_confusion_matrix.png"
)

print(
    OUTPUT_DIR /
    "per_class_metrics.json"
)

print(
    OUTPUT_DIR /
    "error_analysis.json"
)

print(
    OUTPUT_DIR /
    "misclassification_distribution.png"
)

print(
    OUTPUT_DIR /
    "misclassified_predictions.csv"
)

print(
    "\n"
    + "=" * 70
)