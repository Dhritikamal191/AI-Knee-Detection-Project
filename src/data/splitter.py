"""
Dataset Splitter
----------------
Creates reproducible train/validation splits from the
series-level metadata.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

RAW_DATA = Path("data/raw/train_series.csv")
PROCESSED_DIR = Path("data/processed")

TRAIN_OUTPUT = PROCESSED_DIR / "train.csv"
VAL_OUTPUT = PROCESSED_DIR / "val.csv"


class DatasetSplitter:

    def __init__(
        self,
        input_path: Path = RAW_DATA,
        validation_size: float = VALIDATION_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        self.input_path = Path(input_path)
        self.validation_size = validation_size
        self.random_state = random_state

    def load(self) -> pd.DataFrame:

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.input_path}"
            )

        df = pd.read_csv(self.input_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        return df

    def split(self):

        df = self.load()

        train_df, val_df = train_test_split(
            df,
            test_size=self.validation_size,
            random_state=self.random_state,
            shuffle=True,
        )

        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        train_df.to_csv(
            TRAIN_OUTPUT,
            index=False
        )

        val_df.to_csv(
            VAL_OUTPUT,
            index=False
        )

        return train_df, val_df


if __name__ == "__main__":

    splitter = DatasetSplitter()

    train_df, val_df = splitter.split()

    print(f"Total samples : {len(train_df) + len(val_df)}")
    print(f"Training      : {len(train_df)}")
    print(f"Validation    : {len(val_df)}")
    print(f"Train file    : {TRAIN_OUTPUT}")
    print(f"Val file      : {VAL_OUTPUT}")