"""
Dataset Analyzer
----------------
Performs production-oriented integrity checks and
basic target/metadata analysis.
"""

from pathlib import Path

import pandas as pd


TRAIN_FILE = Path("data/processed/train.csv")
VAL_FILE = Path("data/processed/val.csv")


REQUIRED_COLUMNS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
]


class DatasetAnalyzer:

    def __init__(self, train_path=TRAIN_FILE, val_path=VAL_FILE):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)

    def load_datasets(self):

        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {self.train_path}"
            )

        if not self.val_path.exists():
            raise FileNotFoundError(
                f"Validation dataset not found: {self.val_path}"
            )

        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)

        return train_df, val_df

    def validate_schema(self, df):

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

    def analyze(self):

        train_df, val_df = self.load_datasets()

        self.validate_schema(train_df)
        self.validate_schema(val_df)

        print("\n========== DATASET ANALYSIS ==========\n")

        print(f"Training rows   : {len(train_df)}")
        print(f"Validation rows : {len(val_df)}")
        print(f"Total rows      : {len(train_df) + len(val_df)}")

        print("\nColumns:")
        for column in train_df.columns:
            print(f"  - {column}")

        print("\nMissing values — training:")
        print(train_df.isnull().sum())

        print("\nMissing values — validation:")
        print(val_df.isnull().sum())

        print("\nUnique values:")
        for column in train_df.columns:
            print(
                f"{column}: "
                f"{train_df[column].nunique(dropna=True)}"
            )

        print("\nCategorical distributions:")

        for column in [
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        ]:

            print(f"\n--- {column} ---")

            print(
                train_df[column]
                .value_counts(dropna=False)
                .to_string()
            )

        print("\nDuplicate rows:")
        print(
            f"Training   : {train_df.duplicated().sum()}"
        )
        print(
            f"Validation : {val_df.duplicated().sum()}"
        )

        print("\nDuplicate SeriesInstanceUID:")
        print(
            f"Training   : "
            f"{train_df['SeriesInstanceUID'].duplicated().sum()}"
        )
        print(
            f"Validation : "
            f"{val_df['SeriesInstanceUID'].duplicated().sum()}"
        )

        print("\n======================================\n")


if __name__ == "__main__":

    analyzer = DatasetAnalyzer()

    analyzer.analyze()