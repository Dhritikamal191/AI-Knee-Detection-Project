from pathlib import Path
import pandas as pd


class ManifestBuilder:

    def __init__(self, output_path="data/processed/manifest.csv"):

        self.output_path = Path(output_path)

    def build(
        self,
        metadata_path,
        image_root,
        output_path=None
    ):

        metadata_path = Path(metadata_path)
        image_root = Path(image_root)

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        if not image_root.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_root}"
            )

        df = pd.read_csv(metadata_path)

        required_columns = [
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "image_path"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        manifest = df.copy()

        manifest["full_image_path"] = manifest[
            "image_path"
        ].apply(
            lambda x: str(image_root / str(x))
        )

        manifest["image_exists"] = manifest[
            "full_image_path"
        ].apply(
            lambda x: Path(x).exists()
        )

        manifest["dataset_version"] = "v1"

        manifest["status"] = manifest[
            "image_exists"
        ].map(
            {
                True: "valid",
                False: "missing"
            }
        )

        output_path = Path(
            output_path or self.output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        manifest.to_csv(
            output_path,
            index=False
        )

        print("=" * 50)
        print("MANIFEST CREATED")
        print("=" * 50)

        print(f"Total records: {len(manifest)}")
        print(
            f"Valid images: "
            f"{manifest['image_exists'].sum()}"
        )
        print(
            f"Missing images: "
            f"{(~manifest['image_exists']).sum()}"
        )

        print(
            f"Saved to: {output_path}"
        )

        return manifest