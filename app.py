<<<<<<< HEAD
import spaces
import os
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet50
=======
import sys
from pathlib import Path
import tempfile
>>>>>>> 058b6c3 (Finalize Experiment 5 knee OA grading pipeline)

import streamlit as st
from PIL import Image

<<<<<<< HEAD
import gradio as gr
import matplotlib.pyplot as plt
import cv2
from huggingface_hub import hf_hub_download
=======
# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

>>>>>>> 058b6c3 (Finalize Experiment 5 knee OA grading pipeline)

# ============================================================
# IMPORT PROJECT COMPONENTS
# ============================================================

<<<<<<< HEAD
MODEL_PATH = hf_hub_download(
    repo_id="dhriti191/ai-knee-detection-resnet50",
    filename="best_model_experiment3.pt"
)

MODEL_PATH = hf_hub_download(
    repo_id="dhriti191/ai-knee-detection-resnet50",
    filename="best_model_experiment3.pt"
)
=======
from src.inference.predict import predict
from src.explainability.gradcam import generate_gradcam
>>>>>>> 058b6c3 (Finalize Experiment 5 knee OA grading pipeline)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Knee Osteoarthritis Grading",
    page_icon="🦴",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
        margin-bottom: 20px;
    }

    .grade {
        font-size: 55px;
        font-weight: 800;
    }

    .confidence {
        font-size: 22px;
        margin-top: 5px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,193,7,0.4);
        margin-top: 30px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🦴 Knee Osteoarthritis Grading</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-assisted Kellgren–Lawrence grade prediction from knee X-ray images
    using an Ordinal ResNet50 model with Grad-CAM explainability.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        **Final Model**

        Experiment 5

        **Architecture**

        Ordinal ResNet50

        **Output**

        Five osteoarthritis grades:

        - Grade 0
        - Grade 1
        - Grade 2
        - Grade 3
        - Grade 4

        **Explainability**

        Grad-CAM
        """
    )

    st.divider()

    st.write(
        "**Test Performance**"
    )

    st.metric(
        "Accuracy",
        "60.93%"
    )

    st.metric(
        "Macro F1",
        "59.89%"
    )

    st.metric(
        "Quadratic Weighted Kappa",
        "77.07%"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload Knee X-ray</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(
        [1, 1]
    )

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Input X-ray"
        )

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze X-ray",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Running Experiment 5..."
        ):

            # ------------------------------------------------
            # SAVE TEMPORARY IMAGE
            # ------------------------------------------------

            suffix = Path(
                uploaded_file.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:

                tmp.write(
                    uploaded_file.getbuffer()
                )

                temp_path = Path(
                    tmp.name
                )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction_result = predict(
                temp_path
            )

            # ------------------------------------------------
            # GRAD-CAM
            # ------------------------------------------------

            gradcam_result = generate_gradcam(
                temp_path
            )

        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        predicted_grade = (
            prediction_result[
                "prediction"
            ]
        )

        confidence = (
            prediction_result[
                "confidence_percent"
            ]
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "Prediction"
            )

            st.markdown(
                f"""
                <div class="prediction-box">

                <div class="grade">
                Grade {predicted_grade}
                </div>

                <div class="confidence">
                Confidence: {confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # ORDINAL THRESHOLDS
        # ====================================================

        st.markdown(
            '<div class="section-title">Ordinal Threshold Probabilities</div>',
            unsafe_allow_html=True
        )

        threshold_data = (
            prediction_result[
                "threshold_probabilities"
            ]
        )

        threshold_cols = st.columns(
            4
        )

        for i, (
            threshold,
            probability
        ) in enumerate(
            threshold_data.items()
        ):

            with threshold_cols[i]:

                st.metric(
                    threshold,
                    f"{probability * 100:.2f}%"
                )

        # ====================================================
        # GRADE PROBABILITIES
        # ====================================================

        st.markdown(
            '<div class="section-title">Grade Probabilities</div>',
            unsafe_allow_html=True
        )

        grade_data = (
            prediction_result[
                "grade_probabilities"
            ]
        )

        for grade, values in (
            grade_data.items()
        ):

            probability = (
                values[
                    "probability_percent"
                ]
            )

            st.progress(
                min(
                    probability / 100,
                    1.0
                ),
                text=(
                    f"Grade {grade}: "
                    f"{probability:.2f}%"
                )
            )

        # ====================================================
        # GRAD-CAM
        # ====================================================

        st.markdown(
            '<div class="section-title">Grad-CAM Explainability</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            Grad-CAM highlights image regions that contributed
            to the model's prediction.
            """
        )

        # ----------------------------------------------------
        # FIND PREDICTED-GRADE OVERLAY
        # ----------------------------------------------------

        predicted_index = int(
            predicted_grade
        )

        overlay_path = None

        if "classes" in gradcam_result:

            class_info = (
                gradcam_result[
                    "classes"
                ].get(
                    str(predicted_index)
                )
            )

            if class_info:

                overlay_path = Path(
                    class_info[
                        "overlay"
                    ]
                )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if overlay_path is None:

            overlay_path = Path(
                gradcam_result[
                    "overlay"
                ]
            ) if "overlay" in (
                gradcam_result
            ) else None

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        if (
            overlay_path
            and overlay_path.exists()
        ):

            gradcam_col1, gradcam_col2 = (
                st.columns(2)
            )

            with gradcam_col1:

                st.write(
                    "Original"
                )

                st.image(
                    image,
                    use_container_width=True
                )

            with gradcam_col2:

                st.write(
                    f"Grad-CAM — Grade {predicted_grade}"
                )

                st.image(
                    str(overlay_path),
                    use_container_width=True
                )

        else:

            st.warning(
                "Grad-CAM overlay could not be located."
            )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(
            """
            <div class="disclaimer">

            <strong>⚠️ Research / Educational Use Only</strong><br><br>

            This application is an AI research prototype and is
            not a medical device. Predictions should not be used
            for diagnosis, treatment decisions, or clinical
            decision-making. A qualified healthcare professional
            should interpret medical imaging.

<<<<<<< HEAD
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

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
=======
            </div>
            """,
            unsafe_allow_html=True
        )
>>>>>>> 058b6c3 (Finalize Experiment 5 knee OA grading pipeline)
