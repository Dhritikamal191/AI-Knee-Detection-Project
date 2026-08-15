from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validate import load_metadata
from src.utils.config import get_config


LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def create_splits(df: pd.DataFrame, seed: int = 42):
    """
    Create patient/study-level train, validation and test splits.

    The split is performed using StudyInstanceUID so that
    slices belonging to the same MRI study never appear
    across different datasets.
    """

    studies = df.drop_duplicates(
        "StudyInstanceUID"
    ).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        studies,
        test_size=0.30,
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Save dataset splits to the processed-data directory.
    """

    config = get_config()

    train_path = Path(config["data"]["train_csv"])
    val_path = Path(config["data"]["val_csv"])
    test_path = Path(config["data"]["test_csv"])

    train_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    val_df.to_csv(
        val_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(f"Train studies: {len(train_df)}")
    print(f"Validation studies: {len(val_df)}")
    print(f"Test studies: {len(test_df)}")

    print("\nSplits saved successfully.")


if __name__ == "__main__":

    config = get_config()

    df = load_metadata(
        config["data"]["train_csv"]
    )

    train_df, val_df, test_df = create_splits(
        df,
        seed=config["project"]["seed"],
    )

    save_splits(
        train_df,
        val_df,
        test_df,
    )