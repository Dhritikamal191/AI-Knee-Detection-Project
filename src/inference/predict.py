from pathlib import Path
import sys
import hashlib
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from src.models.ordinal_resnet50 import OrdinalResNet50
from src.models.training_config import get_device

# ============================================================
# CONFIG
# ============================================================
def get_model_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

MODEL_PATH = Path(
    hf_hub_download(
        repo_id="dhriti191/ai-knee-detection-resnet50",
        filename="best_model_experiment5.pt"
    )
)

IMAGE_SIZE = 224

NUM_CLASSES = 5

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

    print(f"Using device: {device}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model SHA256: {get_model_hash(MODEL_PATH)}")

    model = OrdinalResNet50(
        num_classes=NUM_CLASSES,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model, device


# ============================================================
# ORDINAL DECODING
# ============================================================

def ordinal_to_grade(
    probabilities
):

    grade = int(
        (
            probabilities >= 0.5
        ).sum()
    )

    return grade


# ============================================================
# GRADE PROBABILITIES
# ============================================================

def calculate_grade_probabilities(
    threshold_probabilities
):

    p1 = threshold_probabilities[0]
    p2 = threshold_probabilities[1]
    p3 = threshold_probabilities[2]
    p4 = threshold_probabilities[3]

    return {

        "0": 1.0 - p1,

        "1": p1 - p2,

        "2": p2 - p3,

        "3": p3 - p4,

        "4": p4
    }


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(
    image_path
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    model, device = load_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(device)

    # --------------------------------------------------------
    # ORDINAL MODEL OUTPUT
    # --------------------------------------------------------

    logits = model(
        tensor
    )

    # Four ordinal threshold probabilities
    threshold_tensor = torch.sigmoid(
        logits
    )[0]

    threshold_probabilities = (
        threshold_tensor
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------
    # PREDICT GRADE
    # --------------------------------------------------------

    predicted_grade = ordinal_to_grade(
        threshold_tensor
    )

    # --------------------------------------------------------
    # GRADE PROBABILITIES
    # --------------------------------------------------------

    grade_probabilities = (
        calculate_grade_probabilities(
            threshold_probabilities
        )
    )

    # Small numerical errors can occasionally
    # produce values such as -0.000001.
    grade_probabilities = {

        grade: max(
            0.0,
            min(
                1.0,
                probability
            )
        )

        for grade, probability
        in grade_probabilities.items()
    }

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------
    #
    # Confidence is the probability assigned
    # to the predicted grade.
    # --------------------------------------------------------

    confidence = grade_probabilities[
        str(predicted_grade)
    ]

    return {

        "prediction":
            CLASS_NAMES[
                predicted_grade
            ],

        "prediction_index":
            predicted_grade,

        "confidence":
            confidence,

        "confidence_percent":
            confidence * 100,

        "threshold_probabilities": {

            "Grade >= 1":
                threshold_probabilities[0],

            "Grade >= 2":
                threshold_probabilities[1],

            "Grade >= 3":
                threshold_probabilities[2],

            "Grade >= 4":
                threshold_probabilities[3]
        },

        "grade_probabilities": {

            grade: {

                "probability":
                    probability,

                "probability_percent":
                    probability * 100

            }

            for grade, probability
            in grade_probabilities.items()
        }
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m "
            "src.inference.predict "
            "<image_path>"
        )

        raise SystemExit

    image_path = sys.argv[1]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Loading Experiment 5..."
    )

    print(
        "=" * 60
    )

    result = predict(
        image_path
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PREDICTION RESULT"
    )

    print(
        "=" * 60
    )

    print(
        f"Predicted Grade: "
        f"{result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence_percent']:.2f}%"
    )

    # --------------------------------------------------------
    # THRESHOLD PROBABILITIES
    # --------------------------------------------------------

    print(
        "\nOrdinal threshold probabilities:"
    )

    for (
        threshold,
        probability
    ) in result[
        "threshold_probabilities"
    ].items():

        print(
            f"{threshold}: "
            f"{probability * 100:.2f}%"
        )

    # --------------------------------------------------------
    # GRADE PROBABILITIES
    # --------------------------------------------------------

    print(
        "\nClass probabilities:"
    )

    for (
        grade,
        values
    ) in result[
        "grade_probabilities"
    ].items():

        print(
            f"Grade {grade}: "
            f"{values['probability_percent']:.2f}%"
        )