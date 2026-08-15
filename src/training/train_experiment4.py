from pathlib import Path
import json
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.data.dataset import get_datasets
from src.models.ordinal_knee_classifier import (
    OrdinalKneeClassifier
)
from src.models.training_config import (
    get_device,
    BATCH_SIZE,
)


# ============================================================
# CONFIG
# ============================================================

BASE_PATH = Path(
    "data/raw/KneeXrayMini"
)

CHECKPOINT_DIR = Path(
    "artifacts/checkpoints"
)

METRICS_DIR = Path(
    "artifacts/metrics"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


NUM_CLASSES = 5

HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 10

HEAD_LR = 1e-4
FINETUNE_LR = 1e-5

WEIGHT_DECAY = 1e-4


# ============================================================
# ORDINAL TARGET
# ============================================================

def create_ordinal_targets(
    labels,
    num_classes=5
):

    targets = []

    for label in labels:

        target = [
            1.0 if label > threshold else 0.0
            for threshold in range(
                num_classes - 1
            )
        ]

        targets.append(target)

    return torch.tensor(
        targets,
        dtype=torch.float32,
        device=labels.device
    )


# ============================================================
# ORDINAL PREDICTION
# ============================================================

def ordinal_predictions(
    outputs
):

    probabilities = torch.sigmoid(
        outputs
    )

    predictions = (
        probabilities > 0.5
    ).sum(dim=1)

    return predictions


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        ordinal_targets = create_ordinal_targets(
            labels,
            NUM_CLASSES
        )

        loss = criterion(
            outputs,
            ordinal_targets
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = ordinal_predictions(
            outputs
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        running_loss / total,
        correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        ordinal_targets = create_ordinal_targets(
            labels,
            NUM_CLASSES
        )

        loss = criterion(
            outputs,
            ordinal_targets
        )

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = ordinal_predictions(
            outputs
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        running_loss / total,
        correct / total
    )


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model,
    learning_rate
):

    parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_accuracy,
    classes
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "num_classes":
            NUM_CLASSES,

        "classes":
            classes
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR /
        "best_model_experiment4.pt"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "EXPERIMENT 4"
    )

    print(
        "ORDINAL RESNET50 CLASSIFICATION"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

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
    # DATA
    # --------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = get_datasets(
        BASE_PATH,
        image_size=224
    )

    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    print(
        f"Classes: "
        f"{train_dataset.classes}"
    )


    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading ImageNet-pretrained ResNet50..."
    )

    model = OrdinalKneeClassifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    ).to(device)


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()


    best_val_loss = float(
        "inf"
    )

    best_val_accuracy = 0.0

    history = []

    start_time = time.time()


    # ========================================================
    # STAGE 1
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "STAGE 1: ORDINAL CLASSIFIER HEAD"
    )

    print("=" * 60)


    optimizer = create_optimizer(
        model,
        HEAD_LR
    )


    for epoch in range(
        HEAD_EPOCHS
    ):

        print(
            f"\nStage 1 - Epoch "
            f"{epoch + 1}/{HEAD_EPOCHS}"
        )

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val Accuracy: "
            f"{val_acc:.4f}"
        )

        history.append({

            "stage": "head",

            "epoch": epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_acc
        })


        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_accuracy = val_acc

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_acc,
                train_dataset.classes
            )

            print(
                "✓ Best Experiment 4 model saved"
            )


    # ========================================================
    # STAGE 2
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "STAGE 2: FINE-TUNING LAYER 3 + LAYER 4"
    )

    print("=" * 60)


    model.unfreeze_last_layers()


    optimizer = create_optimizer(
        model,
        FINETUNE_LR
    )


    for epoch in range(
        FINETUNE_EPOCHS
    ):

        print(
            f"\nStage 2 - Epoch "
            f"{epoch + 1}/{FINETUNE_EPOCHS}"
        )

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val Accuracy: "
            f"{val_acc:.4f}"
        )

        history.append({

            "stage": "fine_tuning",

            "epoch": epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_acc
        })


        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_accuracy = val_acc

            save_checkpoint(
                model,
                optimizer,
                HEAD_EPOCHS + epoch + 1,
                val_loss,
                val_acc,
                train_dataset.classes
            )

            print(
                "✓ Best Experiment 4 model saved"
            )


    # ========================================================
    # METRICS
    # ========================================================

    elapsed = (
        time.time() -
        start_time
    )


    metrics = {

        "experiment": 4,

        "strategy":
            "ordinal_resnet50",

        "best_validation_loss":
            best_val_loss,

        "best_validation_accuracy":
            best_val_accuracy,

        "training_time_seconds":
            elapsed,

        "head_epochs":
            HEAD_EPOCHS,

        "finetune_epochs":
            FINETUNE_EPOCHS,

        "classes":
            train_dataset.classes,

        "history":
            history
    }


    with open(
        METRICS_DIR /
        "experiment4_metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    print("\n")
    print("=" * 60)

    print(
        "EXPERIMENT 4 COMPLETE"
    )

    print("=" * 60)

    print(
        f"Training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        "\nModel:"
    )

    print(
        "artifacts/checkpoints/"
        "best_model_experiment4.pt"
    )

    print(
        "\nMetrics:"
    )

    print(
        "artifacts/metrics/"
        "experiment4_metrics.json"
    )


if __name__ == "__main__":

    main()