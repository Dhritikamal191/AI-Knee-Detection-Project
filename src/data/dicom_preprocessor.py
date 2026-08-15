"""
Production DICOM MRI preprocessing pipeline.

Responsibilities:
- Load DICOM files safely
- Extract pixel arrays
- Apply DICOM rescaling
- Normalize MRI intensities
- Order slices correctly
- Handle missing/corrupt slices
- Extract a fixed number of representative slices
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pydicom


class DICOMPreprocessor:
    """
    Production-oriented preprocessing for MRI DICOM studies.
    """

    def __init__(
        self,
        target_size: int = 224,
        num_slices: int = 8,
    ):
        self.target_size = target_size
        self.num_slices = num_slices

    # ---------------------------------------------------------
    # DICOM LOADING
    # ---------------------------------------------------------

    @staticmethod
    def load_dicom(path: Path) -> Optional[pydicom.dataset.FileDataset]:
        """
        Safely load a DICOM file.
        """

        try:
            return pydicom.dcmread(
                str(path),
                force=True,
            )

        except Exception:
            return None

    # ---------------------------------------------------------
    # PIXEL EXTRACTION
    # ---------------------------------------------------------

    @staticmethod
    def get_pixel_array(
        dicom: pydicom.dataset.FileDataset,
    ) -> Optional[np.ndarray]:
        """
        Extract and convert DICOM pixel data.
        """

        try:
            image = dicom.pixel_array.astype(np.float32)

            # Apply DICOM rescaling where available
            slope = float(
                getattr(dicom, "RescaleSlope", 1.0)
            )

            intercept = float(
                getattr(dicom, "RescaleIntercept", 0.0)
            )

            image = image * slope + intercept

            return image

        except Exception:
            return None

    # ---------------------------------------------------------
    # INTENSITY NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Robust percentile-based MRI normalization.
        """

        image = image.astype(np.float32)

        # Remove invalid values
        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        non_zero = image[image > 0]

        if non_zero.size == 0:
            return np.zeros_like(image, dtype=np.float32)

        low = np.percentile(non_zero, 1)
        high = np.percentile(non_zero, 99)

        if high <= low:
            return np.zeros_like(image, dtype=np.float32)

        image = np.clip(
            image,
            low,
            high,
        )

        image = (image - low) / (high - low)

        return image.astype(np.float32)

    # ---------------------------------------------------------
    # SLICE ORDERING
    # ---------------------------------------------------------

    @staticmethod
    def slice_position(
        dicom: pydicom.dataset.FileDataset,
    ) -> float:
        """
        Determine slice position for ordering.
        """

        if hasattr(dicom, "ImagePositionPatient"):
            try:
                return float(
                    dicom.ImagePositionPatient[2]
                )
            except Exception:
                pass

        if hasattr(dicom, "SliceLocation"):
            try:
                return float(
                    dicom.SliceLocation
                )
            except Exception:
                pass

        if hasattr(dicom, "InstanceNumber"):
            try:
                return float(
                    dicom.InstanceNumber
                )
            except Exception:
                pass

        return 0.0

    # ---------------------------------------------------------
    # LOAD STUDY
    # ---------------------------------------------------------

    def load_study(
        self,
        study_path: Path,
    ) -> List[np.ndarray]:
        """
        Load all valid slices from a DICOM study.
        """

        study_path = Path(study_path)

        if not study_path.exists():
            raise FileNotFoundError(
                f"Study directory not found: {study_path}"
            )

        slices = []

        for file_path in study_path.rglob("*"):

            if not file_path.is_file():
                continue

            dicom = self.load_dicom(file_path)

            if dicom is None:
                continue

            image = self.get_pixel_array(dicom)

            if image is None:
                continue

            if image.ndim != 2:
                continue

            position = self.slice_position(dicom)

            slices.append(
                (
                    position,
                    image,
                )
            )

        if not slices:
            raise ValueError(
                f"No valid DICOM slices found in {study_path}"
            )

        # Correct anatomical ordering
        slices.sort(
            key=lambda x: x[0]
        )

        return [
            image
            for _, image in slices
        ]

    # ---------------------------------------------------------
    # RESIZE
    # ---------------------------------------------------------

    def resize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Resize image to model input resolution.
        """

        import cv2

        return cv2.resize(
            image,
            (
                self.target_size,
                self.target_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

    # ---------------------------------------------------------
    # REPRESENTATIVE SLICE SELECTION
    # ---------------------------------------------------------

    def select_slices(
        self,
        slices: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Select a fixed number of representative slices.

        Uses evenly distributed sampling rather than
        simply taking the first N slices.
        """

        total = len(slices)

        if total == 0:
            raise ValueError(
                "Cannot select slices from empty study."
            )

        if total >= self.num_slices:

            indices = np.linspace(
                0,
                total - 1,
                self.num_slices,
                dtype=int,
            )

            selected = [
                slices[i]
                for i in indices
            ]

        else:

            # Repeat the final valid slice if the study
            # contains fewer slices than required.
            selected = list(slices)

            while len(selected) < self.num_slices:
                selected.append(slices[-1])

        return selected

    # ---------------------------------------------------------
    # COMPLETE PIPELINE
    # ---------------------------------------------------------

    def preprocess_study(
        self,
        study_path: Path,
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.

        Returns:
            numpy array with shape:

            (num_slices, target_size, target_size)
        """

        slices = self.load_study(
            study_path
        )

        selected = self.select_slices(
            slices
        )

        processed = []

        for image in selected:

            image = self.normalize(
                image
            )

            image = self.resize(
                image
            )

            processed.append(
                image
            )

        return np.stack(
            processed,
            axis=0,
        ).astype(np.float32)