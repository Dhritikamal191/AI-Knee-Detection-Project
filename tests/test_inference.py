from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.predict import predict


def create_test_image(tmp_path: Path) -> Path:
    """
    Create a synthetic RGB image for inference testing.

    This keeps CI independent of the local training/validation dataset.
    """

    image_array = np.zeros(
        (224, 224, 3),
        dtype=np.uint8
    )

    # Add a simple non-uniform structure so the image
    # is not completely blank.
    image_array[60:170, 70:150] = 128

    image = Image.fromarray(
        image_array,
        mode="RGB"
    )

    image_path = tmp_path / "test_knee_xray.png"

    image.save(image_path)

    return image_path


def test_prediction_returns_result(tmp_path):
    test_image = create_test_image(tmp_path)

    result = predict(test_image)

    assert result is not None
    assert isinstance(result, dict)


def test_prediction_grade_is_valid(tmp_path):
    test_image = create_test_image(tmp_path)

    result = predict(test_image)

    grade = result["prediction_index"]

    assert grade in [0, 1, 2, 3, 4]


def test_confidence_is_valid(tmp_path):
    test_image = create_test_image(tmp_path)

    result = predict(test_image)

    confidence = result["confidence"]

    assert 0.0 <= confidence <= 1.0


def test_grade_probabilities_are_valid(tmp_path):
    test_image = create_test_image(tmp_path)

    result = predict(test_image)

    probabilities = result["grade_probabilities"]

    assert len(probabilities) == 5

    for grade, values in probabilities.items():

        probability = values["probability"]

        assert 0.0 <= probability <= 1.0


def test_grade_probabilities_sum_to_one(tmp_path):
    test_image = create_test_image(tmp_path)

    result = predict(test_image)

    probabilities = result["grade_probabilities"]

    total = sum(
        values["probability"]
        for values in probabilities.values()
    )

    assert abs(total - 1.0) < 1e-5
