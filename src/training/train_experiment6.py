from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)

from src.data.dataset import get_datasets
from src.models.ordinal_resnet50 import OrdinalResNet50
from src.models.training_config import (
    get_device,
    BATCH_SIZE
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
# CONFIG
# ============================================================

NUM_CLASSES = 5

HEAD_EPOCHS = 5

FINETUNE_EPOCHS = 20

HEAD_LR = 3e-4

LAYER2_LR = 1e-6
LAYER3_LR = 3e-6
LAYER4_LR = 1e-5
CLASSIFIER_LR = 3e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 6

IMAGE_SIZE = 224


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        256
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.90, 1.0),
        ratio=(0.95, 1.05)
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),

    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12
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


val_transform = transforms.Compose([

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
# ORDINAL TARGET
# ============================================================

def create_ordinal_targets(
    labels,
    num_classes=5
):

    """
    Converts:

        Grade 0 -> [0,0,0,0]
        Grade 1 -> [1,0,0,0]
        Grade 2 -> [1,1,0,0]
        Grade 3 -> [1,1,1,0]
        Grade 4 -> [1,1,1,1]
    """

    targets = []

    for label in labels:

        target = [
            1.0 if label >= threshold
            else 0.0

            for threshold in range(
                1,
                num_classes
            )
        ]

        targets.append(
            target
        )

    return torch.tensor(
        targets,
        dtype=torch.float32
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

    # Number of thresholds passed
    grades = (
        probabilities >= 0.5
    ).sum(
        dim=1
    )

    return grades.long()


# ============================================================
# ORDINAL POSITIVE WEIGHTS
# ============================================================

def calculate_pos_weights(
    dataset
):

    print(
        "\nExperiment 6 ordinal threshold weights:"
    )

    # --------------------------------------------------------
    # Controlled threshold-specific weighting
    #
    # Experiment 5 used:
    #
    # >=1 : ~0.81
    # >=2 : ~1.17
    # >=3 : ~2.28
    # >=4 : ~5.69
    #
    # The very high >=4 weight contributed to aggressive
    # Grade-4 predictions and overconfident Grade-3 -> Grade-4
    # errors.
    #
    # Experiment 6 uses moderated weights.
    # --------------------------------------------------------

    weights = [
        1.00,   # Grade >= 1
        1.15,   # Grade >= 2
        1.75,   # Grade >= 3
        2.50    # Grade >= 4
    ]

    for threshold, weight in zip(
        range(1, NUM_CLASSES),
        weights
    ):

        print(
            f"Grade >= {threshold}: "
            f"pos_weight={weight:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )

# ============================================================
# LOSS
# ============================================================

class OrdinalLoss(nn.Module):

    def __init__(
        self,
        pos_weight
    ):

        super().__init__()

        self.register_buffer(
            "pos_weight",
            pos_weight
        )

    def forward(
        self,
        logits,
        targets
    ):

        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight
        )


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

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        ordinal_targets = (
            create_ordinal_targets(
                labels,
                NUM_CLASSES
            ).to(device)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            ordinal_targets
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = ordinal_to_grade(
            logits.detach()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    loss = (
        total_loss /
        len(loader.dataset)
    )

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

    qwk = cohen_kappa_score(
        all_labels,
        all_predictions,
        weights="quadratic"
    )

    return (
        loss,
        accuracy,
        macro_f1,
        qwk
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

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        ordinal_targets = (
            create_ordinal_targets(
                labels,
                NUM_CLASSES
            ).to(device)
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            ordinal_targets
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = ordinal_to_grade(
            logits
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    loss = (
        total_loss /
        len(loader.dataset)
    )

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

    balanced_acc = (
        balanced_accuracy_score(
            all_labels,
            all_predictions
        )
    )

    qwk = cohen_kappa_score(
        all_labels,
        all_predictions,
        weights="quadratic"
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "qwk": qwk,
        "predictions": all_predictions,
        "labels": all_labels
    }


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model,
    stage
):

    if stage == "head":

        return torch.optim.AdamW(
            model.backbone.fc.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY
        )

    # --------------------------------------------------------
    # Discriminative learning rates
    # --------------------------------------------------------

    return torch.optim.AdamW(

        [

            {
                "params":
                    model.backbone.layer2.parameters(),

                "lr":
                    LAYER2_LR
            },

            {
                "params":
                    model.backbone.layer3.parameters(),

                "lr":
                    LAYER3_LR
            },

            {
                "params":
                    model.backbone.layer4.parameters(),

                "lr":
                    LAYER4_LR
            },

            {
                "params":
                    model.backbone.fc.parameters(),

                "lr":
                    CLASSIFIER_LR
            }

        ],

        weight_decay=WEIGHT_DECAY
    )


# ============================================================
# SAVE
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    classes,
    class_to_idx
):

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "metrics":
            metrics,

        "num_classes":
            NUM_CLASSES,

        "classes":
            classes,

        "class_to_idx":
            class_to_idx,

        "model_type":
            "ordinal_resnet50",

        "ordinal_thresholds":
            [
                ">=1",
                ">=2",
                ">=3",
                ">=4"
            ]
    }

    torch.save(

        checkpoint,

        CHECKPOINT_DIR /
        "best_model_experiment6.pt"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 6"
    )

    print(
        "ORDINAL RESNET50 + MODERATE BALANCING + "
        "MEDICAL-SAFE AUGMENTATION"
    )

    print(
        "=" * 70
    )

    start_time = time.time()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()

    print(
        f"\nUsing device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
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
        image_size=IMAGE_SIZE
    )

    # --------------------------------------------------------
    # Replace transforms
    #
    # This assumes the datasets expose the torchvision-style
    # .transform property, which your existing dataset appears
    # to use.
    # --------------------------------------------------------

    if hasattr(
        train_dataset,
        "transform"
    ):

        train_dataset.transform = (
            train_transform
        )

    if hasattr(
        val_dataset,
        "transform"
    ):

        val_dataset.transform = (
            val_transform
        )

    if hasattr(
        test_dataset,
        "transform"
    ):

        test_dataset.transform = (
            val_transform
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

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # POSITIVE WEIGHTS
    # --------------------------------------------------------

    pos_weight = (
        calculate_pos_weights(
            train_dataset
        ).to(device)
    )

    criterion = OrdinalLoss(
        pos_weight
    ).to(device)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading ImageNet-pretrained ResNet50..."
    )

    model = OrdinalResNet50(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(device)

    # ========================================================
    # STAGE 1
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE 1: CLASSIFIER HEAD"
    )

    print(
        "=" * 70
    )

    model.freeze_backbone()

    optimizer = create_optimizer(
        model,
        "head"
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )

    best_qwk = -1.0

    best_metrics = None

    history = []

    for epoch in range(
        HEAD_EPOCHS
    ):

        print(
            f"\nStage 1 - Epoch "
            f"{epoch + 1}/{HEAD_EPOCHS}"
        )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )

        train_loss, train_acc, train_f1, train_qwk = (
            train_metrics
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Train QWK: {train_qwk:.4f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Balanced Acc: "
            f"{val_metrics['balanced_accuracy']:.4f} | "
            f"Val QWK: {val_metrics['qwk']:.4f}"
        )

        scheduler.step(
            val_metrics["qwk"]
        )

        record = {

            "stage": "head",

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "train_macro_f1":
                train_f1,

            "train_qwk":
                train_qwk,

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_macro_f1":
                val_metrics["macro_f1"],

            "val_balanced_accuracy":
                val_metrics["balanced_accuracy"],

            "val_qwk":
                val_metrics["qwk"]
        }

        history.append(
            record
        )

        if (
            val_metrics["qwk"]
            > best_qwk
        ):

            best_qwk = (
                val_metrics["qwk"]
            )

            best_metrics = (
                val_metrics
            )

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_metrics,
                train_dataset.classes,
                train_dataset.class_to_idx
            )

            print(
                "✓ Best QWK model saved"
            )

    # ========================================================
    # STAGE 2
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE 2: LAYER 2 + 3 + 4 FINE-TUNING"
    )

    print(
        "=" * 70
    )

    model.unfreeze_layers_2_to_4()

    optimizer = create_optimizer(
        model,
        "finetune"
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )

    epochs_without_improvement = 0

    for epoch in range(
        FINETUNE_EPOCHS
    ):

        print(
            f"\nStage 2 - Epoch "
            f"{epoch + 1}/{FINETUNE_EPOCHS}"
        )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )

        train_loss, train_acc, train_f1, train_qwk = (
            train_metrics
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Train QWK: {train_qwk:.4f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Balanced Acc: "
            f"{val_metrics['balanced_accuracy']:.4f} | "
            f"Val QWK: {val_metrics['qwk']:.4f}"
        )

        scheduler.step(
            val_metrics["qwk"]
        )

        record = {

            "stage":
                "fine_tuning",

            "epoch":
                HEAD_EPOCHS + epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "train_macro_f1":
                train_f1,

            "train_qwk":
                train_qwk,

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_macro_f1":
                val_metrics["macro_f1"],

            "val_balanced_accuracy":
                val_metrics["balanced_accuracy"],

            "val_qwk":
                val_metrics["qwk"]
        }

        history.append(
            record
        )

        if (
            val_metrics["qwk"]
            > best_qwk
        ):

            best_qwk = (
                val_metrics["qwk"]
            )

            best_metrics = (
                val_metrics
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                HEAD_EPOCHS + epoch + 1,
                val_metrics,
                train_dataset.classes,
                train_dataset.class_to_idx
            )

            print(
                "✓ Best QWK model saved"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No QWK improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

            if (
                epochs_without_improvement
                >= PATIENCE
            ):

                print(
                    "\nEarly stopping."
                )

                break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_DIR /
        "best_model_experiment6.pt",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # TEST
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    test_metrics = validate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"\nTest Loss: "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test QWK: "
        f"{test_metrics['qwk']:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )

    report = classification_report(

        test_metrics["labels"],

        test_metrics["predictions"],

        target_names=[
            str(x)
            for x in train_dataset.classes
        ],

        zero_division=0
    )

    print(
        report
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(

        test_metrics["labels"],

        test_metrics["predictions"]
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    final_metrics = {

        "experiment": 5,

        "strategy":
            "ordinal_resnet50_"
            "moderate_balancing_"
            "medical_safe_augmentation",

        "best_validation_qwk":
            best_qwk,

        "test_loss":
            test_metrics["loss"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_macro_f1":
            test_metrics["macro_f1"],

        "test_balanced_accuracy":
            test_metrics[
                "balanced_accuracy"
            ],

        "test_qwk":
            test_metrics["qwk"],

        "confusion_matrix":
            cm.tolist(),

        "history":
            history,

        "training_time_seconds":
            elapsed,

        "classes":
            train_dataset.classes,

        "class_to_idx":
            train_dataset.class_to_idx,

        "ordinal_thresholds":
            [
                ">=1",
                ">=2",
                ">=3",
                ">=4"
            ]
    }

    with open(

        METRICS_DIR /
        "experiment6_metrics.json",

        "w"
    ) as f:

        json.dump(
            final_metrics,
            f,
            indent=4
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPERIMENT 6 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Best Validation QWK: "
        f"{best_qwk:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test QWK: "
        f"{test_metrics['qwk']:.4f}"
    )

    print(
        "\nModel:"
    )

    print(
        "artifacts/checkpoints/"
        "best_model_experiment6.pt"
    )

    print(
        "\nMetrics:"
    )

    print(
        "artifacts/metrics/"
        "experiment6_metrics.json"
    )


if __name__ == "__main__":

    main()