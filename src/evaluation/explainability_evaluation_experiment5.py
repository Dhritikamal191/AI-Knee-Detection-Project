from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.explainability.gradcam import (
    load_model,
    GradCAM,
    get_grade_target
)


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = Path(
    "data/raw/KneeXrayMini/test"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
NUM_CLASSES = 5
BATCH_SIZE = 32

CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]

# Number of examples from each category
NUM_CORRECT = 5
NUM_INCORRECT = 5


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
# ORDINAL PROBABILITIES
# ============================================================

def ordinal_to_class_probabilities(
    output
):

    probabilities = torch.sigmoid(
        output
    )

    p_ge_1 = probabilities[0, 0].item()
    p_ge_2 = probabilities[0, 1].item()
    p_ge_3 = probabilities[0, 2].item()
    p_ge_4 = probabilities[0, 3].item()

    class_probabilities = np.array([

        1.0 - p_ge_1,

        p_ge_1 - p_ge_2,

        p_ge_2 - p_ge_3,

        p_ge_3 - p_ge_4,

        p_ge_4

    ])

    class_probabilities = np.clip(
        class_probabilities,
        0.0,
        1.0
    )

    total = class_probabilities.sum()

    if total > 0:

        class_probabilities /= total

    return class_probabilities


# ============================================================
# ORDINAL CONFIDENCE
# ============================================================

def get_prediction_confidence(
    probabilities,
    prediction
):

    if prediction == 0:

        return float(
            1.0 - probabilities[0]
        )

    elif prediction == 4:

        return float(
            probabilities[4]
        )

    else:

        return float(
            min(
                probabilities[prediction],
                probabilities[prediction - 1]
                + probabilities[prediction]
            )
        )


# ============================================================
# SAVE IMAGE
# ============================================================

def save_cam_image(
    cam,
    original_image,
    output_path
):

    import cv2

    cam = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        cam,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    original = np.array(
        original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    Image.fromarray(
        heatmap
    ).save(
        output_path / "heatmap.jpg"
    )

    Image.fromarray(
        overlay
    ).save(
        output_path / "overlay.jpg"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPERIMENT 5 EXPLAINABILITY EVALUATION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, device = load_model()

    print(
        f"\nUsing device: {device}"
    )


    # ========================================================
    # LOAD TEST DATASET
    # ========================================================

    print(
        "\nLoading test dataset..."
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    print(
        f"Classes: "
        f"{test_dataset.classes}"
    )


    # ========================================================
    # DATA LOADER
    # ========================================================

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    # ========================================================
    # TARGET LAYER
    # ========================================================

    target_layer = (
        model.backbone.layer4[-1]
    )

    gradcam = GradCAM(
        model,
        target_layer
    )


    # ========================================================
    # GENERATE PREDICTIONS
    # ========================================================

    print(
        "\nGenerating test predictions..."
    )

    all_results = []

    global_index = 0

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device
            )

            outputs = model(
                images
            )

            threshold_probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                threshold_probabilities >= 0.5
            ).sum(
                dim=1
            ).long()

            for batch_index in range(
                images.size(0)
            ):

                prediction = int(
                    predictions[
                        batch_index
                    ].item()
                )

                true_label = int(
                    labels[
                        batch_index
                    ].item()
                )

                output = outputs[
                    batch_index:
                    batch_index + 1
                ]

                class_probabilities = (
                    ordinal_to_class_probabilities(
                        output
                    )
                )

                confidence = float(
                    class_probabilities[
                        prediction
                    ]
                )

                correct = (
                    prediction ==
                    true_label
                )

                image_path = Path(
                    test_dataset.samples[
                        global_index
                    ][0]
                )

                all_results.append({

                    "index":
                        global_index,

                    "image":
                        str(image_path),

                    "true_grade":
                        true_label,

                    "predicted_grade":
                        prediction,

                    "correct":
                        correct,

                    "confidence":
                        confidence,

                    "class_probabilities":
                        class_probabilities.tolist()
                })

                global_index += 1


    print(
        f"Predictions generated: "
        f"{len(all_results)}"
    )


    # ========================================================
    # SEPARATE CORRECT / INCORRECT
    # ========================================================

    correct_results = [

        x for x in all_results

        if x["correct"]
    ]

    incorrect_results = [

        x for x in all_results

        if not x["correct"]
    ]


    print(
        f"\nCorrect predictions: "
        f"{len(correct_results)}"
    )

    print(
        f"Incorrect predictions: "
        f"{len(incorrect_results)}"
    )


    # ========================================================
    # SELECT REPRESENTATIVE EXAMPLES
    # ========================================================

    # Highest-confidence examples are useful for
    # demonstrating strong model decisions.

    correct_results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    incorrect_results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )


    selected_correct = (
        correct_results[
            :NUM_CORRECT
        ]
    )

    selected_incorrect = (
        incorrect_results[
            :NUM_INCORRECT
        ]
    )


    selected = [

        ("correct", x)

        for x in selected_correct

    ] + [

        ("incorrect", x)

        for x in selected_incorrect

    ]


    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    summary = []


    for category, result in selected:

        index = result["index"]

        image_path = Path(
            result["image"]
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Image: {image_path.name}"
        )

        print(
            f"True Grade: "
            f"{result['true_grade']}"
        )

        print(
            f"Predicted Grade: "
            f"{result['predicted_grade']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )


        # ----------------------------------------------------
        # OUTPUT DIRECTORY
        # ----------------------------------------------------

        sample_dir = (

            OUTPUT_DIR
            /
            category
            /
            f"sample_{index:04d}"
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # LOAD ORIGINAL IMAGE
        # ----------------------------------------------------

        original_image = Image.open(
            image_path
        ).convert("RGB")


        original_image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ).save(
            sample_dir /
            "original.jpg"
        )


        # ----------------------------------------------------
        # IMAGE TENSOR
        # ----------------------------------------------------

        image_tensor = transform(
            original_image
        ).unsqueeze(
            0
        ).to(
            device
        )


        # ----------------------------------------------------
        # FRESH FORWARD PASS
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

        output = model(
            image_tensor
        )


        prediction = result[
            "predicted_grade"
        ]


        # ----------------------------------------------------
        # TARGET PREDICTED GRADE
        # ----------------------------------------------------

        target = get_grade_target(

            output,

            prediction
        )


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        _, cam = gradcam.generate(

            image_tensor,

            target
        )


        # ----------------------------------------------------
        # SAVE CAM
        # ----------------------------------------------------

        save_cam_image(

            cam,

            original_image,

            sample_dir
        )


        # ----------------------------------------------------
        # SAMPLE METADATA
        # ----------------------------------------------------

        sample_summary = {

            "index":
                index,

            "category":
                category,

            "image":
                str(image_path),

            "true_grade":
                result["true_grade"],

            "predicted_grade":
                prediction,

            "correct":
                result["correct"],

            "confidence":
                result["confidence"],

            "confidence_percent":
                result["confidence"] * 100,

            "class_probabilities":
                result["class_probabilities"],

            "original":
                str(
                    sample_dir /
                    "original.jpg"
                ),

            "heatmap":
                str(
                    sample_dir /
                    "heatmap.jpg"
                ),

            "overlay":
                str(
                    sample_dir /
                    "overlay.jpg"
                )
        }


        summary.append(
            sample_summary
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_file = (
        OUTPUT_DIR /
        "explainability_summary.json"
    )


    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {

                "experiment":
                    "Experiment 5",

                "test_samples":
                    len(all_results),

                "correct_predictions":
                    len(correct_results),

                "incorrect_predictions":
                    len(incorrect_results),

                "accuracy":
                    len(correct_results)
                    /
                    len(all_results),

                "gradcam_samples":
                    len(summary),

                "samples":
                    summary

            },
            f,
            indent=4
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPLAINABILITY EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nSummary:"
    )

    print(
        summary_file
    )


if __name__ == "__main__":

    main()