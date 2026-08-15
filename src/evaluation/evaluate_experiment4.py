from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data.dataset import get_datasets
from src.models.ordinal_knee_classifier import (
    OrdinalKneeClassifier
)
from src.models.training_config import (
    get_device,
    BATCH_SIZE,
)

import json


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(
    "data/raw/KneeXrayMini"
)

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/"
    "best_model_experiment4.pt"
)

OUTPUT_PATH = Path(
    "artifacts/metrics/"
    "experiment4_test_metrics.json"
)


NUM_CLASSES = 5


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 50)
    print("EXPERIMENT 4 TEST EVALUATION")
    print("=" * 50)

    device = get_device()

    print(
        f"\nUsing device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(
        "\nLoading test dataset..."
    )

    (
        _,
        _,
        test_dataset
    ) = get_datasets(
        BASE_PATH,
        image_size=224
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    print(
        f"Classes: "
        f"{test_dataset.classes}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading Experiment 4 model..."
    )

    model = OrdinalKneeClassifier(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False
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

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nRunning predictions..."
    )

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities > 0.5
            ).sum(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=[
            "0",
            "1",
            "2",
            "3",
            "4"
        ],
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4]
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("=" * 32)
    print("EXPERIMENT 4 TEST RESULTS")
    print("=" * 32)

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

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(report)

    print(
        "Confusion Matrix:"
    )

    print(cm)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    metrics = {

        "experiment": 4,

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "weighted_f1":
            float(f1),

        "macro_f1":
            float(macro_f1),

        "classification_report":
            report,

        "confusion_matrix":
            cm.tolist(),

        "checkpoint_epoch":
            checkpoint["epoch"]
    }

    with open(
        OUTPUT_PATH,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        "\n✓ Test metrics saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()