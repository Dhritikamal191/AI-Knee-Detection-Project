from pathlib import Path
import json

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from src.data.dataset import get_datasets
from src.models.knee_classifier import KneeClassifier


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path("data/raw/KneeXrayMini")

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/best_model_experiment3.pt"
)

METRICS_DIR = Path(
    "artifacts/metrics"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading test dataset...")

_, _, test_dataset = get_datasets(
    BASE_PATH,
    image_size=224
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(
    f"Test images: {len(test_dataset)}"
)

print(
    f"Classes: {test_dataset.classes}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = KneeClassifier(
    num_classes=5,
    pretrained=False,
    freeze_backbone=True
).to(device)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint['epoch']}"
)


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []

print("\nRunning predictions...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )
)


print("\n================================")
print("TEST RESULTS")
print("================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "test_accuracy": float(
        accuracy
    ),

    "weighted_precision": float(
        precision
    ),

    "weighted_recall": float(
        recall
    ),

    "weighted_f1": float(
        f1
    ),

    "classes":
        test_dataset.classes,

    "confusion_matrix":
        cm.tolist(),

    "classification_report":
        classification_report(
            all_labels,
            all_predictions,
            target_names=test_dataset.classes,
            output_dict=True,
            zero_division=0
        )
}


with open(
    METRICS_DIR /
    "experiment3_test_metrics.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\n✓ Test metrics saved to:"
)

print(
    "artifacts/metrics/test_metrics.json"
)