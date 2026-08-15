"""
Dataset validation and sample inspection.

Validates:
- Study paths
- DICOM loading
- Number of slices
- Tensor shape
- Label shape
- NaN/Inf values
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import KneeMRIDataset


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_PATH = Path("data")

TRAIN_CSV = Path("data/train.csv")

NUM_SLICES = 8
IMAGE_SIZE = 224


# ---------------------------------------------------------
# VALIDATE DATAFRAME
# ---------------------------------------------------------

def validate_dataframe(df: pd.DataFrame):

    print("\n========== DATAFRAME VALIDATION ==========")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "StudyInstanceUID" not in df.columns:
        raise ValueError(
            "StudyInstanceUID column is missing."
        )

    duplicate_count = df["StudyInstanceUID"].duplicated().sum()

    print(
        f"Duplicate studies: {duplicate_count:,}"
    )

    missing_ids = df["StudyInstanceUID"].isna().sum()

    print(
        f"Missing StudyInstanceUID: {missing_ids:,}"
    )


# ---------------------------------------------------------
# VALIDATE SAMPLE
# ---------------------------------------------------------

def validate_sample(
    dataset: KneeMRIDataset,
    index: int = 0,
):

    print("\n========== SAMPLE VALIDATION ==========")

    image, label = dataset[index]

    print(
        f"Image shape: {tuple(image.shape)}"
    )

    print(
        f"Label shape: {tuple(label.shape)}"
    )

    print(
        f"Image dtype: {image.dtype}"
    )

    print(
        f"Label dtype: {label.dtype}"
    )

    # Expected:
    # [1, 8, 224, 224]

    expected_shape = (
        1,
        NUM_SLICES,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(image.shape) != expected_shape:

        raise ValueError(
            f"Unexpected image shape: "
            f"{tuple(image.shape)}. "
            f"Expected {expected_shape}"
        )

    if torch.isnan(image).any():

        raise ValueError(
            "Image contains NaN values."
        )

    if torch.isinf(image).any():

        raise ValueError(
            "Image contains infinite values."
        )

    if torch.isnan(label).any():

        raise ValueError(
            "Labels contain NaN values."
        )

    if torch.isinf(label).any():

        raise ValueError(
            "Labels contain infinite values."
        )

    print("Image values:")
    print(
        f"  Min: {image.min().item():.4f}"
    )
    print(
        f"  Max: {image.max().item():.4f}"
    )
    print(
        f"  Mean: {image.mean().item():.4f}"
    )

    print(
        f"Labels: {label.tolist()}"
    )

    print("\nSample validation PASSED.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV}"
        )

    df = pd.read_csv(
        TRAIN_CSV
    )

    validate_dataframe(df)

    dataset = KneeMRIDataset(
        dataframe=df,
        base_path=BASE_PATH,
        num_slices=NUM_SLICES,
        target_size=IMAGE_SIZE,
    )

    print(
        f"\nDataset size: {len(dataset):,}"
    )

    validate_sample(
        dataset,
        index=0,
    )

    print(
        "\n========================================"
    )
    print(
        "DATA PIPELINE VALIDATION PASSED"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()