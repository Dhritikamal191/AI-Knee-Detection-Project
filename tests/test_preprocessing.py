from pathlib import Path

from PIL import Image

from src.inference.predict import transform


def test_image_transform():

    image = Image.new(
        "RGB",
        (512, 512),
        color="white"
    )

    tensor = transform(image)

    assert tensor.shape == (3, 224, 224)