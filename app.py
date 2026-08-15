import spaces

import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet50

from PIL import Image

import gradio as gr
import matplotlib.pyplot as plt
import cv2


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "artifacts/checkpoints/best_model_experiment3.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = ["0", "1", "2", "3", "4"]


# ============================================================
# MODEL
# ============================================================

class KneeClassifier(nn.Module):

    def __init__(
        self,
        num_classes=5,
        dropout=0.3
    ):
        super().__init__()

        self.backbone = resnet50(
            weights=None
        )

        input_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Sequential(

            nn.Linear(
                input_features,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(f"Using device: {DEVICE}")
print("Loading Experiment 3 model...")


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = KneeClassifier(
    num_classes=checkpoint.get(
        "num_classes",
        5
    )
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint.get('epoch')}"
)

print(
    f"Validation accuracy: "
    f"{checkpoint.get('val_accuracy')}"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
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
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
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

        self.activations = output.detach()

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0].detach()
        )

    def generate(
        self,
        input_tensor,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        score = output[
            0,
            class_index
        ]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():

            cam = (
                cam - cam.min()
            ) / (
                cam.max() - cam.min()
            )

        else:

            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


target_layer = (
    model.backbone.layer4[-1]
)

gradcam = GradCAM(
    model,
    target_layer
)


# ============================================================
# GRAD-CAM VISUALIZATION
# ============================================================

def create_gradcam(
    original_image,
    cam
):

    image = np.array(
        original_image.convert("RGB")
    )

    height, width = image.shape[:2]

    cam_resized = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(
        255 * cam_resized
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        0.55 * image +
        0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return (
        Image.fromarray(heatmap),
        Image.fromarray(overlay)
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    if image is None:

        return (
            "Please upload a knee X-ray.",
            "—",
            "",
            None,
            None,
            None
        )

    image = image.convert("RGB")

    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    predicted_class = int(
        torch.argmax(
            probabilities,
            dim=1
        ).item()
    )

    confidence = float(
        probabilities[
            0,
            predicted_class
        ].item()
    )

    predicted_grade = (
        CLASSES[predicted_class]
    )

    probability_values = (
        probabilities[0]
        .cpu()
        .numpy()
        * 100
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    cam_input = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    cam = gradcam.generate(
        cam_input,
        predicted_class
    )

    heatmap, overlay = (
        create_gradcam(
            image,
            cam
        )
    )

    # --------------------------------------------------------
    # Probability chart
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        CLASSES,
        probability_values
    )

    ax.set_xlabel(
        "Knee Grade"
    )

    ax.set_ylabel(
        "Probability (%)"
    )

    ax.set_title(
        "Model Prediction Probabilities"
    )

    ax.set_ylim(
        0,
        100
    )

    for bar, value in zip(
        bars,
        probability_values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 1,

            f"{value:.1f}%",

            ha="center"
        )

    fig.tight_layout()

    fig.canvas.draw()

    chart = np.asarray(
        fig.canvas.buffer_rgba()
    )[:, :, :3]

    plt.close(fig)

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    prediction_text = (
        f"## Predicted Grade: "
        f"{predicted_grade}"
    )

    confidence_text = (
        f"## Confidence: "
        f"{confidence * 100:.2f}%"
    )

    probability_text = "\n".join(

        [
            f"**Grade {grade}:** "
            f"{prob:.2f}%"

            for grade, prob
            in zip(
                CLASSES,
                probability_values
            )
        ]
    )

    return (
        prediction_text,
        confidence_text,
        probability_text,
        chart,
        heatmap,
        overlay
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="AI Knee Detection"
) as demo:

    gr.Markdown(
        """
# 🦵 AI Knee Detection

### Deep Learning-Based Knee X-ray Classification

This application uses an **ImageNet-pretrained ResNet50**
fine-tuned for five-class knee grading.

**Features**

- 🧠 ResNet50 deep learning classification
- 📊 Prediction confidence
- 📈 Class probability distribution
- 🔥 Grad-CAM explainability
- 🩻 Knee X-ray image analysis

> ⚠️ **Research/Educational Use Only**
>
> This application is not a medical diagnostic tool and
> should not be used for clinical decision-making.
        """
    )

    with gr.Row():

        image_input = gr.Image(
            type="pil",
            label="Upload Knee X-ray"
        )

        with gr.Column():

            prediction = gr.Markdown()

            confidence = gr.Markdown()

            probabilities = gr.Markdown()

    predict_button = gr.Button(
        "🔍 Analyze X-ray",
        variant="primary"
    )

    gr.Markdown(
        "## 📊 Prediction Probabilities"
    )

    probability_chart = gr.Image(
        label="Class Probability Distribution"
    )

    gr.Markdown(
        "## 🔥 Grad-CAM Explainability"
    )

    gr.Markdown(
        """
The Grad-CAM visualizations highlight image regions
that contributed to the model's prediction.
        """
    )

    with gr.Row():

        heatmap_output = gr.Image(
            label="Grad-CAM Heatmap"
        )

        overlay_output = gr.Image(
            label="Grad-CAM Overlay"
        )

    predict_button.click(
        fn=predict,

        inputs=image_input,

        outputs=[
            prediction,
            confidence,
            probabilities,
            probability_chart,
            heatmap_output,
            overlay_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch()