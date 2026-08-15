from pathlib import Path

import numpy as np
import torch
import cv2

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

OUTPUT_DIR = Path(
    "artifacts/gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# TRANSFORM
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
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        target = output[
            0,
            class_index
        ]

        target.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = torch.relu(
            cam
        )

        cam = torch.nn.functional.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return output, cam

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    original_image,
    cam
):

    original = np.array(
        original_image
    )

    original = cv2.resize(
        original,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return heatmap, overlay


# ============================================================
# MAIN
# ============================================================

def generate_gradcam(
    image_path
):

    print(
        "\nLoading Experiment 3 model..."
    )

    model, device = load_model()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # TARGET LAYER
    # --------------------------------------------------------

    target_layer = (
        model.backbone.layer4[-1].conv3
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    model.zero_grad()

    output = model(
        image_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    prediction = (
        probabilities.argmax(
            dim=1
        ).item()
    )

    confidence = (
        probabilities[
            0,
            prediction
        ].item()
    )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    _, cam = gradcam.generate(
        image_tensor,
        prediction
    )

    gradcam.close()

    # --------------------------------------------------------
    # CREATE VISUALS
    # --------------------------------------------------------

    heatmap, overlay = create_overlay(
        original_image,
        cam
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_base = (
        OUTPUT_DIR /
        Path(image_path).stem
    )

    original_path = (
        output_base /
        "original.jpg"
    )

    heatmap_path = (
        output_base /
        "heatmap.jpg"
    )

    overlay_path = (
        output_base /
        "gradcam_overlay.jpg"
    )

    output_base.mkdir(
        parents=True,
        exist_ok=True
    )

    original_resized = original_image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    original_resized.save(
        original_path
    )

    Image.fromarray(
        heatmap
    ).save(
        heatmap_path
    )

    Image.fromarray(
        overlay
    ).save(
        overlay_path
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print(
        "\n================================"
    )

    print(
        "GRAD-CAM RESULT"
    )

    print(
        "================================"
    )

    print(
        f"Predicted Grade: "
        f"{CLASS_NAMES[prediction]}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        "\nGenerated files:"
    )

    print(
        original_path
    )

    print(
        heatmap_path
    )

    print(
        overlay_path
    )

    return {
        "prediction":
            CLASS_NAMES[prediction],

        "confidence":
            confidence,

        "original":
            str(original_path),

        "heatmap":
            str(heatmap_path),

        "overlay":
            str(overlay_path)
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m src.explainability.gradcam "
            "<image_path>"
        )

        raise SystemExit

    image_path = sys.argv[1]

    generate_gradcam(
        image_path
    )