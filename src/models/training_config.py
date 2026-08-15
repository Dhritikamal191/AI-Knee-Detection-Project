import torch
import torch.nn as nn


def build_loss(class_weights=None):
    """
    Build weighted cross-entropy loss.

    class_weights:
        Optional tensor containing one weight per class.
    """

    if class_weights is not None:
        class_weights = class_weights.float()

    return nn.CrossEntropyLoss(
        weight=class_weights
    )


def build_optimizer(
    model,
    learning_rate=1e-4,
    weight_decay=1e-4
):
    """
    AdamW optimizer for stable fine-tuning.
    """

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    return torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )


def build_scheduler(
    optimizer,
    factor=0.5,
    patience=2
):
    """
    Reduce learning rate when validation
    performance stops improving.
    """

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience
    )

IMAGE_SIZE = 224

BATCH_SIZE = 2

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

def get_device():
    """
    Automatically select GPU when available.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_training_components(
    model,
    class_weights=None,
    learning_rate=1e-4,
    weight_decay=1e-4
):

    device = get_device()

    loss_fn = build_loss(
        class_weights
    )

    optimizer = build_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = build_scheduler(
        optimizer
    )

    return {
        "device": device,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "scheduler": scheduler
    }