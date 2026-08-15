from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.knee_classifier import KneeClassifier
from src.models.training_config import get_device


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "artifacts/checkpoints/best_model_experiment3.pt"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    device = get_device()

    model = KneeClassifier(
        num_classes=5,
        pretrained=False,
        freeze_backbone=False
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, device


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(image_path):

    model, device = load_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    outputs = model(tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = (
        prediction.item()
    )

    confidence_value = (
        confidence.item()
    )

    probability_values = (
        probabilities[0]
        .cpu()
        .tolist()
    )

    return {
        "prediction": CLASS_NAMES[
            predicted_class
        ],

        "confidence":
            confidence_value,

        "probabilities":
            {
                CLASS_NAMES[i]:
                    probability_values[i]
                for i in range(5)
            }
    }


# ============================================================
# TEST FROM COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m src.inference.predict "
            "<image_path>"
        )

        raise SystemExit

    image_path = sys.argv[1]

    print(
        "\nLoading Experiment 3 model..."
    )

    result = predict(
        image_path
    )

    print(
        "\n================================"
    )

    print(
        "PREDICTION"
    )

    print(
        "================================"
    )

    print(
        f"Predicted Grade: "
        f"{result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        "\nClass Probabilities:"
    )

    for class_name, probability in (
        result["probabilities"].items()
    ):

        print(
            f"Grade {class_name}: "
            f"{probability * 100:.2f}%"
        )