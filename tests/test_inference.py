from pathlib import Path

from src.inference.predict import predict


TEST_IMAGE = Path(
    "data/raw/KneeXrayMini/val/2/9986207L.png"
)


def test_prediction_returns_result():

    assert TEST_IMAGE.exists(), (
        f"Test image not found: {TEST_IMAGE}"
    )

    result = predict(
        TEST_IMAGE
    )

    assert isinstance(
        result,
        dict
    )


def test_prediction_grade_is_valid():

    result = predict(
        TEST_IMAGE
    )

    assert result["prediction_index"] in {
        0, 1, 2, 3, 4
    }


def test_confidence_is_valid():

    result = predict(
        TEST_IMAGE
    )

    confidence = result[
        "confidence"
    ]

    assert 0.0 <= confidence <= 1.0


def test_grade_probabilities_are_valid():

    result = predict(
        TEST_IMAGE
    )

    probabilities = {
        grade: values["probability"]
        for grade, values
        in result[
            "grade_probabilities"
        ].items()
    }

    assert len(probabilities) == 5

    for probability in probabilities.values():

        assert 0.0 <= probability <= 1.0


def test_grade_probabilities_sum_to_one():

    result = predict(
        TEST_IMAGE
    )

    total = sum(
        values["probability"]
        for values
        in result[
            "grade_probabilities"
        ].values()
    )

    assert abs(total - 1.0) < 1e-5