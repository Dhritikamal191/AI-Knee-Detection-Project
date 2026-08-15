from pathlib import Path
import pandas as pd


class DatasetValidator:

    REQUIRED_COLUMNS = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "image_path"
    ]

    def __init__(self, metadata_path: str, image_root: str):

        self.metadata_path = Path(metadata_path)
        self.image_root = Path(image_root)

    def validate_metadata(self):

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        df = pd.read_csv(self.metadata_path)

        print(f"Metadata rows: {len(df)}")

        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        duplicate_count = df["StudyInstanceUID"].duplicated().sum()

        print(f"Duplicate studies: {duplicate_count}")

        missing_values = df[self.REQUIRED_COLUMNS].isnull().sum()

        print("\nMissing values:")
        print(missing_values)

        return df

    def validate_images(self, df):

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_root}"
            )

        missing_images = 0
        existing_images = 0

        for image_path in df["image_path"]:

            full_path = self.image_root / str(image_path)

            if full_path.exists():
                existing_images += 1
            else:
                missing_images += 1

        print(f"\nExisting images: {existing_images}")
        print(f"Missing images: {missing_images}")

        if missing_images > 0:
            print(
                "WARNING: Some metadata entries do not have "
                "corresponding image files."
            )

        return missing_images

    def run(self):

        print("=" * 50)
        print("DATASET VALIDATION")
        print("=" * 50)

        df = self.validate_metadata()

        self.validate_images(df)

        print("\nDataset validation completed.")

        return df