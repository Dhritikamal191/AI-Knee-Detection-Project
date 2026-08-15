from pathlib import Path

from torchvision import datasets, transforms


def get_transforms(image_size=224):

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def get_datasets(
    base_path,
    image_size=224
):

    base_path = Path(base_path)

    train_transform, eval_transform = get_transforms(
        image_size
    )

    train_dataset = datasets.ImageFolder(
        base_path / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        base_path / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        base_path / "test",
        transform=eval_transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )