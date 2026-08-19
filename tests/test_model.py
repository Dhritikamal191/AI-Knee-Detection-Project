import torch

from src.models.ordinal_resnet50 import OrdinalResNet50


def test_model_creation():

    model = OrdinalResNet50(
        num_classes=5,
        pretrained=False
    )

    assert model is not None


def test_model_output_shape():

    model = OrdinalResNet50(
        num_classes=5,
        pretrained=False
    )

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    with torch.no_grad():

        output = model(
            dummy_input
        )

    assert output.shape == (
        1,
        4
    )