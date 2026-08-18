from pathlib import Path
import sys
import json

import numpy as np
import torch
import cv2

from PIL import Image
from torchvision import transforms

from src.training.train_experiment5 import OrdinalResNet50
from src.models.training_config import get_device


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "artifacts/checkpoints/"
    "best_model_experiment5.pt"
)

OUTPUT_DIR = Path(
    "artifacts/gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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

    print(
        f"Using device: {device}"
    )

    model = OrdinalResNet50(
        num_classes=NUM_CLASSES,
        pretrained=False
    ).to(device)

    # --------------------------------------------------------
    # PyTorch 2.6+
    #
    # weights_only=False is required because the checkpoint
    # contains objects such as NumPy scalars.
    #
    # Use this only with your own/trusted checkpoint.
    # --------------------------------------------------------

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
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = (
            target_layer
        )

        self.activations = None

        self.gradients = None

        # ----------------------------------------------------
        # FORWARD HOOK
        # ----------------------------------------------------

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        # ----------------------------------------------------
        # BACKWARD HOOK
        # ----------------------------------------------------

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    # ========================================================
    # ACTIVATION
    # ========================================================

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    # ========================================================
    # GRADIENT
    # ========================================================

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0]
        )


    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    def generate(
        self,
        image_tensor,
        target
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        output = self.model(
            image_tensor
        )

        # ----------------------------------------------------
        # BACKWARD
        #
        # target is a scalar tensor representing the ordinal
        # target we want to explain.
        # ----------------------------------------------------

        target.backward(
            retain_graph=True
        )

        gradients = (
            self.gradients
        )

        activations = (
            self.activations
        )

        if gradients is None:

            raise RuntimeError(
                "Gradients were not captured. "
                "Check the Grad-CAM target layer."
            )

        if activations is None:

            raise RuntimeError(
                "Activations were not captured. "
                "Check the Grad-CAM target layer."
            )

        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # WEIGHT ACTIVATIONS
        # ----------------------------------------------------

        cam = (
            weights *
            activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # REMOVE NEGATIVE ACTIVATIONS
        # ----------------------------------------------------

        cam = torch.relu(
            cam
        )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        cam = torch.nn.functional.interpolate(

            cam,

            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            mode="bilinear",

            align_corners=False
        )

        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        cam = (
            cam
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:

            cam /= maximum

        return output, cam


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    cam
):

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

    return heatmap


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

    heatmap = create_heatmap(
        cam
    )

    overlay = cv2.addWeighted(

        original,

        0.60,

        heatmap,

        0.40,

        0
    )

    return heatmap, overlay


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    array,
    path
):

    Image.fromarray(
        array
    ).save(
        path,
        quality=95
    )


# ============================================================
# ORDINAL GRADE TARGET
# ============================================================

def get_grade_target(
    output,
    grade
):
    """
    Creates a Grad-CAM target for an individual
    knee osteoarthritis grade.

    Experiment 5 uses 4 ordinal outputs:

        output[0, 0] -> Grade >= 1
        output[0, 1] -> Grade >= 2
        output[0, 2] -> Grade >= 3
        output[0, 3] -> Grade >= 4

    For Grad-CAM we create a target corresponding
    to each individual grade.
    """

    if grade == 0:

        # Grade 0:
        # NOT Grade >= 1

        return -output[0, 0]


    elif grade == 1:

        # Grade 1:
        # >= 1 but NOT >= 2

        return (
            output[0, 0]
            -
            output[0, 1]
        )


    elif grade == 2:

        # Grade 2:
        # >= 2 but NOT >= 3

        return (
            output[0, 1]
            -
            output[0, 2]
        )


    elif grade == 3:

        # Grade 3:
        # >= 3 but NOT >= 4

        return (
            output[0, 2]
            -
            output[0, 3]
        )


    elif grade == 4:

        # Grade 4:
        # >= 4

        return output[0, 3]


    else:

        raise ValueError(
            f"Invalid grade: {grade}"
        )


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path
):

    print(
        "\nLoading Experiment 5 model..."
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, device = load_model()


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    original_image = Image.open(
        image_path
    ).convert("RGB")


    # ========================================================
    # TRANSFORM IMAGE
    # ========================================================

    image_tensor = transform(
        original_image
    ).unsqueeze(
        0
    ).to(
        device
    )


    # ========================================================
    # TARGET LAYER
    # ========================================================
    #
    # Experiment 5 uses ResNet50.
    #
    # The last convolutional block is:
    #
    # model.backbone.layer4[-1]
    #
    # This is the correct target for Grad-CAM.
    # ========================================================

    target_layer = (
        model.backbone.layer4[-1]
    )


    gradcam = GradCAM(
        model,
        target_layer
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    model.zero_grad(
        set_to_none=True
    )


    output = model(
        image_tensor
    )


    # ========================================================
    # EXPERIMENT 5 ORDINAL OUTPUTS
    # ========================================================
    #
    # output shape:
    #
    # [1, 4]
    #
    # output[0,0] = Grade >= 1
    # output[0,1] = Grade >= 2
    # output[0,2] = Grade >= 3
    # output[0,3] = Grade >= 4
    # ========================================================

    threshold_probabilities = (
        torch.sigmoid(
            output
        )
    )


    threshold_values = (

        threshold_probabilities[
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )


    # ========================================================
    # PREDICT GRADE
    # ========================================================

    prediction = int(

        (
            threshold_probabilities[
                0
            ]
            >= 0.5
        )
        .sum()
        .item()
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if prediction == 0:

        confidence = float(

            1.0
            -
            threshold_probabilities[
                0,
                0
            ].item()
        )


    elif prediction == 4:

        confidence = float(

            threshold_probabilities[
                0,
                3
            ].item()
        )


    else:

        lower_threshold = (

            threshold_probabilities[
                0,
                prediction - 1
            ].item()
        )

        upper_threshold = (

            threshold_probabilities[
                0,
                prediction
            ].item()
        )

        confidence = float(

            min(

                lower_threshold,

                1.0
                -
                upper_threshold
            )
        )


    # ========================================================
    # CONVERT ORDINAL THRESHOLDS
    # TO 5-CLASS PROBABILITIES
    # ========================================================

    p_ge_1 = float(
        threshold_values[0]
    )

    p_ge_2 = float(
        threshold_values[1]
    )

    p_ge_3 = float(
        threshold_values[2]
    )

    p_ge_4 = float(
        threshold_values[3]
    )


    # --------------------------------------------------------
    # INDIVIDUAL GRADE PROBABILITIES
    # --------------------------------------------------------

    class_probabilities = np.array([

        # Grade 0
        1.0 - p_ge_1,

        # Grade 1
        p_ge_1 - p_ge_2,

        # Grade 2
        p_ge_2 - p_ge_3,

        # Grade 3
        p_ge_3 - p_ge_4,

        # Grade 4
        p_ge_4

    ], dtype=np.float64)


    # ========================================================
    # NUMERICAL SAFETY
    # ========================================================

    class_probabilities = np.clip(

        class_probabilities,

        0.0,

        1.0
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    probability_sum = (
        class_probabilities.sum()
    )


    if probability_sum > 0:

        class_probabilities = (

            class_probabilities
            /
            probability_sum
        )


    # ========================================================
    # PRINT ORDINAL PROBABILITIES
    # ========================================================

    print(
        "\nOrdinal threshold probabilities:"
    )


    for i, probability in enumerate(
        threshold_values
    ):

        print(

            f"Grade >= {i + 1}: "
            f"{probability * 100:.2f}%"
        )


    # ========================================================
    # PRINT PREDICTION
    # ========================================================

    print(

        f"\nPredicted Grade: "
        f"{CLASS_NAMES[prediction]}"
    )


    print(

        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )


    # ========================================================
    # PRINT CLASS PROBABILITIES
    # ========================================================

    print(
        "\nClass probabilities:"
    )


    for i in range(
        NUM_CLASSES
    ):

        print(

            f"Grade {i}: "
            f"{class_probabilities[i] * 100:.2f}%"
        )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_base = (

        OUTPUT_DIR
        /
        image_path.stem
    )


    output_base.mkdir(

        parents=True,

        exist_ok=True
    )


    # ========================================================
    # SAVE ORIGINAL
    # ========================================================

    original_resized = (

        original_image.resize(

            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )
    )


    original_path = (

        output_base
        /
        "original.jpg"
    )


    original_resized.save(

        original_path,

        quality=95
    )


    # ========================================================
    # GENERATE GRAD-CAM FOR ALL GRADES
    # ========================================================

    class_results = {}


    for class_index in range(
        NUM_CLASSES
    ):

        print(

            f"Generating Grad-CAM "
            f"for Grade {class_index}..."
        )


        # ----------------------------------------------------
        # CREATE FRESH MODEL OUTPUT
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )


        output_for_cam = model(
            image_tensor
        )


        # ----------------------------------------------------
        # CREATE TARGET FOR THIS GRADE
        # ----------------------------------------------------

        target = get_grade_target(

            output_for_cam,

            class_index
        )


        # ----------------------------------------------------
        # GENERATE CAM
        # ----------------------------------------------------

        _, cam = gradcam.generate(

            image_tensor,

            target
        )


        # ----------------------------------------------------
        # CREATE VISUALIZATION
        # ----------------------------------------------------

        heatmap, overlay = (

            create_overlay(

                original_image,

                cam
            )
        )


        # ----------------------------------------------------
        # PATHS
        # ----------------------------------------------------

        heatmap_path = (

            output_base
            /
            f"grade_{class_index}_heatmap.jpg"
        )


        overlay_path = (

            output_base
            /
            f"grade_{class_index}_overlay.jpg"
        )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_image(

            heatmap,

            heatmap_path
        )


        save_image(

            overlay,

            overlay_path
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        class_results[
            str(class_index)
        ] = {

            "grade":
                CLASS_NAMES[class_index],

            "probability":
                float(
                    class_probabilities[
                        class_index
                    ]
                ),

            "probability_percent":
                float(
                    class_probabilities[
                        class_index
                    ]
                    * 100
                ),

            "heatmap":
                str(
                    heatmap_path
                ),

            "overlay":
                str(
                    overlay_path
                )
        }


    # ========================================================
    # CLOSE GRAD-CAM
    # ========================================================

    gradcam.close()


    # ========================================================
    # THRESHOLD RESULTS
    # ========================================================

    threshold_results = {

        f"Grade >= {i + 1}":

            float(
                threshold_values[i]
            )

        for i in range(
            len(threshold_values)
        )
    }


    # ========================================================
    # SAVE PREDICTION JSON
    # ========================================================

    result = {

        "prediction":
            CLASS_NAMES[prediction],

        "prediction_index":
            int(prediction),

        "confidence":
            float(confidence),

        "confidence_percent":
            float(
                confidence * 100
            ),

        "threshold_probabilities":
            threshold_results,

        "grade_probabilities": {

            CLASS_NAMES[i]: {

                "probability":
                    float(
                        class_probabilities[i]
                    ),

                "probability_percent":
                    float(
                        class_probabilities[i]
                        * 100
                    )
            }

            for i in range(
                NUM_CLASSES
            )
        },

        "classes":
            class_results,

        "original":
            str(
                original_path
            )
    }


    # ========================================================
    # WRITE JSON
    # ========================================================

    json_path = (

        output_base
        /
        "prediction.json"
    )


    with open(

        json_path,

        "w"
    ) as f:

        json.dump(

            result,

            f,

            indent=4
        )


    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "GRAD-CAM RESULT"
    )

    print(
        "=" * 60
    )


    print(

        f"Predicted Grade: "
        f"{CLASS_NAMES[prediction]}"
    )


    print(

        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )


    # ========================================================
    # PRINT CLASS PROBABILITIES
    # ========================================================

    print(
        "\nClass probabilities:"
    )


    for i in range(
        NUM_CLASSES
    ):

        print(

            f"Grade {i}: "
            f"{class_probabilities[i] * 100:.2f}%"
        )


    # ========================================================
    # GENERATED DIRECTORY
    # ========================================================

    print(
        "\nGenerated directory:"
    )

    print(
        output_base
    )


    # ========================================================
    # PREDICTION JSON
    # ========================================================

    print(
        "\nPrediction JSON:"
    )

    print(
        json_path
    )


    # ========================================================
    # GENERATED FILES
    # ========================================================

    print(
        "\nFiles generated:"
    )


    print(
        "Original:"
    )

    print(
        original_path
    )


    for i in range(
        NUM_CLASSES
    ):

        print(
            f"\nGrade {i} Heatmap:"
        )

        print(

            class_results[
                str(i)
            ][
                "heatmap"
            ]
        )


        print(
            f"Grade {i} Overlay:"
        )

        print(

            class_results[
                str(i)
            ][
                "overlay"
            ]
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return result


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
            "src.explainability.gradcam "
            "<image_path>"
        )

        raise SystemExit


    image_path = sys.argv[1]


    generate_gradcam(
        image_path
    )