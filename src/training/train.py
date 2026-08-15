from pathlib import Path
import json
import time

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from src.data.dataset import get_datasets
from src.models.knee_classifier import KneeClassifier
from src.models.training_config import (
    get_device,
    BATCH_SIZE,
)


# ============================================================
# PATHS
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


# ============================================================
# EXPERIMENT 3 CONFIG
# ============================================================

NUM_CLASSES = 5

HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 10

HEAD_LR = 1e-4
FINETUNE_LR = 1e-5

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05


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

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
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

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
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
# MODERATE BALANCED SAMPLER
# ============================================================

def create_balanced_sampler(dataset):

    labels = [
        label
        for _, label in dataset.samples
    ]

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long
    )

    class_counts = torch.bincount(
        labels_tensor,
        minlength=NUM_CLASSES
    ).float()

    print("\nTraining class counts:")

    for i, count in enumerate(class_counts):

        print(
            f"Class {i}: "
            f"{int(count.item())}"
        )

    # --------------------------------------------------------
    # Square-root inverse frequency
    #
    # This is intentionally moderate.
    # It gives minority classes more exposure without
    # forcing every class to have exactly equal probability.
    # --------------------------------------------------------

    class_weights = (
        1.0 /
        torch.sqrt(class_counts)
    )

    sample_weights = torch.tensor(
        [
            class_weights[label].item()
            for label in labels
        ],
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print("\nSampler class weights:")

    for i, weight in enumerate(class_weights):

        print(
            f"Class {i}: "
            f"{weight.item():.6f}"
        )

    return sampler


# ============================================================
# FREEZE BACKBONE
# ============================================================

def freeze_backbone(model):

    for parameter in model.backbone.parameters():

        parameter.requires_grad = False

    for parameter in model.backbone.fc.parameters():

        parameter.requires_grad = True


# ============================================================
# UNFREEZE LAST RESNET BLOCKS
# ============================================================

def unfreeze_last_blocks(model):

    for parameter in model.backbone.layer3.parameters():

        parameter.requires_grad = True

    for parameter in model.backbone.layer4.parameters():

        parameter.requires_grad = True

    for parameter in model.backbone.fc.parameters():

        parameter.requires_grad = True


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model,
    learning_rate
):

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    return torch.optim.AdamW(
        trainable_parameters,
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
    classes,
    class_to_idx,
    stage
):

    checkpoint = {

        "epoch": epoch,

        "stage": stage,

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
            classes,

        "class_to_idx":
            class_to_idx
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR /
        "best_model_experiment3.pt"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "EXPERIMENT 3"
    )

    print(
        "MODERATE BALANCING + LABEL SMOOTHING"
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
    # SAMPLER
    # --------------------------------------------------------

    sampler = create_balanced_sampler(
        train_dataset
    )


    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
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

    model = KneeClassifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    ).to(device)


    # --------------------------------------------------------
    # LABEL-SMOOTHED LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )


    best_val_loss = float(
        "inf"
    )

    history = []

    start_time = time.time()


    # ========================================================
    # STAGE 1
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "STAGE 1: CLASSIFIER HEAD"
    )

    print("=" * 60)


    freeze_backbone(
        model
    )

    optimizer = create_optimizer(
        model,
        HEAD_LR
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
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

        scheduler.step(
            val_loss
        )

        lr = optimizer.param_groups[0]["lr"]

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

        print(
            f"Learning Rate: "
            f"{lr:.2e}"
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
                val_acc,

            "learning_rate":
                lr
        })


        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_acc,
                train_dataset.classes,
                train_dataset.class_to_idx,
                "head"
            )

            print(
                "✓ Best Experiment 3 model saved"
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


    unfreeze_last_blocks(
        model
    )

    optimizer = create_optimizer(
        model,
        FINETUNE_LR
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
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

        scheduler.step(
            val_loss
        )

        lr = optimizer.param_groups[0]["lr"]

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

        print(
            f"Learning Rate: "
            f"{lr:.2e}"
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
                val_acc,

            "learning_rate":
                lr
        })


        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                HEAD_EPOCHS + epoch + 1,
                val_loss,
                val_acc,
                train_dataset.classes,
                train_dataset.class_to_idx,
                "fine_tuning"
            )

            print(
                "✓ Best Experiment 3 model saved"
            )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    elapsed = (
        time.time() -
        start_time
    )

    best_accuracy = max(
        item["val_accuracy"]
        for item in history
    )


    metrics = {

        "experiment": 3,

        "strategy":
            "moderate_balanced_sampler_label_smoothing",

        "label_smoothing":
            LABEL_SMOOTHING,

        "best_validation_loss":
            best_val_loss,

        "best_validation_accuracy":
            best_accuracy,

        "training_time_seconds":
            elapsed,

        "head_epochs":
            HEAD_EPOCHS,

        "finetune_epochs":
            FINETUNE_EPOCHS,

        "classes":
            train_dataset.classes,

        "class_to_idx":
            train_dataset.class_to_idx,

        "history":
            history
    }


    with open(
        METRICS_DIR /
        "experiment3_metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)

    print(
        "EXPERIMENT 3 COMPLETE"
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
        f"{best_accuracy:.4f}"
    )

    print(
        "\nModel:"
    )

    print(
        "artifacts/checkpoints/"
        "best_model_experiment3.pt"
    )

    print(
        "\nMetrics:"
    )

    print(
        "artifacts/metrics/"
        "experiment3_metrics.json"
    )


if __name__ == "__main__":

    main()