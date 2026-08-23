from pathlib import Path
from typing import Any
import json
import tempfile
import inspect

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.inference.predict import predict
from src.utils.config import get_model_config


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Knee OA Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CONFIG
# ============================================================

GRADE_NAMES = [
    "Grade 0",
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4",
]

GRADE_DESCRIPTIONS = {
    0: "No radiographic evidence of osteoarthritis",
    1: "Doubtful / early osteoarthritic changes",
    2: "Mild osteoarthritis",
    3: "Moderate osteoarthritis",
    4: "Severe osteoarthritis",
}

MODEL_SHA256 = (
    "f2200b43966dce1498e47ad6ee45cb35e5cec831246b7667323e16fd9d7e1667"
)


# ============================================================
# LOAD MODEL CONFIG
# ============================================================

try:
    MODEL_CONFIG = get_model_config()
except Exception:
    MODEL_CONFIG = {}


MODEL_INFO = MODEL_CONFIG.get("model", {})
PERFORMANCE = MODEL_CONFIG.get("performance", {})


MODEL_NAME = MODEL_INFO.get(
    "name",
    "ordinal_resnet50",
)

MODEL_ARCHITECTURE = MODEL_INFO.get(
    "architecture",
    "OrdinalResNet50",
)

MODEL_EXPERIMENT = MODEL_INFO.get(
    "experiment",
    "experiment5",
)

MODEL_REPOSITORY = (
    MODEL_INFO
    .get("repository", {})
    .get(
        "repo_id",
        "dhriti191/ai-knee-detection-resnet50",
    )
)

IMAGE_SIZE = (
    MODEL_INFO
    .get("input", {})
    .get(
        "image_size",
        224,
    )
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 5% 0%,
                rgba(99, 102, 241, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 95% 5%,
                rgba(59, 130, 246, 0.08),
                transparent 28%
            ),
            #08090d;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* ======================================================
       TYPOGRAPHY
       ====================================================== */

    h1,
    h2,
    h3,
    h4 {
        letter-spacing: -0.035em !important;
    }

    h1 {
        font-weight: 850 !important;
    }

    h2 {
        font-weight: 800 !important;
    }

    h3 {
        font-weight: 750 !important;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b0d13 0%,
                #090a0e 100%
            );

        border-right:
            1px solid rgba(255, 255, 255, 0.07);
    }

    section[data-testid="stSidebar"] * {
        color: #d7d9e0;
    }

    /* ======================================================
       NATIVE METRIC CARDS
       ====================================================== */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(99, 102, 241, 0.09),
                rgba(255, 255, 255, 0.02)
            );

        border:
            1px solid rgba(255, 255, 255, 0.08);

        border-radius: 18px;

        padding: 20px;

        min-height: 125px;

        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.12);
    }

    div[data-testid="stMetric"]:hover {
        border-color:
            rgba(129, 140, 248, 0.28);
    }

    div[data-testid="stMetricLabel"] {
        color: #858a98 !important;
        font-size: 0.75rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f4f4f5 !important;
        font-weight: 800 !important;
    }

    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background:
            rgba(255, 255, 255, 0.018);

        border:
            1px dashed rgba(255, 255, 255, 0.13);

        border-radius: 18px;

        padding: 8px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color:
            rgba(129, 140, 248, 0.40);
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 11px;

        min-height: 44px;

        background:
            linear-gradient(
                135deg,
                rgba(99, 102, 241, 0.18),
                rgba(59, 130, 246, 0.10)
            );

        border:
            1px solid rgba(129, 140, 248, 0.25);

        color: #eef0ff;

        font-weight: 700;

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            background 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);

        border-color:
            rgba(129, 140, 248, 0.55);

        background:
            linear-gradient(
                135deg,
                rgba(99, 102, 241, 0.28),
                rgba(59, 130, 246, 0.17)
            );

        color: white;
    }

    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #a5b4fc;
    }

    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        border:
            1px solid rgba(255, 255, 255, 0.08);

        border-radius: 14px;

        overflow: hidden;
    }

    /* ======================================================
       ALERTS
       ====================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ======================================================
       EXPANDERS
       ====================================================== */

    details {
        border-radius: 14px !important;
        border-color:
            rgba(255, 255, 255, 0.08) !important;
    }

    /* ======================================================
       CODE
       ====================================================== */

    pre {
        border-radius: 12px !important;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer-line {
        margin-top: 60px;
        padding-top: 25px;
        border-top:
            1px solid rgba(255, 255, 255, 0.07);

        text-align: center;

        color: #626775;

        font-size: 0.75rem;

        line-height: 1.7;
    }

    /* ======================================================
       STATUS
       ====================================================== */

    .status-box {
        padding: 12px 15px;

        border-radius: 12px;

        background:
            rgba(52, 211, 153, 0.06);

        border:
            1px solid rgba(52, 211, 153, 0.14);

        color: #a7f3d0;

        font-size: 0.82rem;
    }

    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 900px) {

        .block-container {
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
    }

    @media (max-width: 600px) {

        .block-container {
            padding-top: 1rem;
        }

        div[data-testid="stMetric"] {
            min-height: 105px;
            padding: 15px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_value(
    result: Any,
    *keys,
    default=None,
):
    """
    Safely retrieve a value from either:
    - dictionary
    - object with attributes
    """

    if result is None:
        return default

    if isinstance(result, dict):

        for key in keys:

            if key in result:
                return result[key]

    else:

        for key in keys:

            if hasattr(result, key):
                return getattr(result, key)

    return default


def to_percentage(value):
    """
    Convert probability to percentage.

    Supports:
        0.82 -> 82
        82 -> 82
    """

    if value is None:
        return 0.0

    try:
        value = float(value)
    except Exception:
        return 0.0

    if 0 <= value <= 1:
        return value * 100

    return value


def extract_grade(result):
    """
    Extract predicted grade from the prediction result.
    """

    value = get_value(
        result,
        "predicted_grade",
        "prediction",
        "grade",
        "predicted_class",
        "class_id",
        default=0,
    )

    try:
        return int(value)
    except Exception:
        return 0


def extract_confidence(result):
    """
    Extract prediction confidence.
    """

    value = get_value(
        result,
        "confidence",
        "prediction_confidence",
        "max_probability",
        default=0,
    )

    return to_percentage(value)


def extract_probabilities(result):
    """
    Extract class probabilities.

    Supports common result formats.
    """

    values = get_value(
        result,
        "class_probabilities",
        "grade_probabilities",
        "probabilities",
        "probs",
        default=None,
    )

    if values is None:
        return None

    if isinstance(values, dict):

        probabilities = []

        for grade in GRADE_NAMES:

            value = (
                values.get(grade)
                or values.get(
                    grade.replace(
                        "Grade ",
                        "",
                    )
                )
                or 0
            )

            probabilities.append(
                to_percentage(value)
            )

        return probabilities

    try:

        return [
            to_percentage(value)
            for value in values
        ]

    except Exception:

        return None


def save_uploaded_file(uploaded_file):
    """
    Save uploaded image temporarily.
    """

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:

        tmp.write(
            uploaded_file.getbuffer()
        )

        return Path(tmp.name)


def normalize_result_for_json(result):
    """
    Convert prediction result into a JSON-safe object.
    """

    if isinstance(result, dict):
        return result

    if hasattr(result, "__dict__"):
        return result.__dict__

    return str(result)


# ============================================================
# GRAD-CAM HELPERS
# ============================================================

def find_gradcam_function():
    """
    Import Grad-CAM function without making the entire app
    fail if the implementation is unavailable.
    """

    try:

        from src.explainability.gradcam import (
            generate_gradcam,
        )

        return generate_gradcam

    except Exception:

        return None


def run_gradcam(
    image_path,
    predicted_grade,
):
    """
    Attempt to work with the existing Grad-CAM implementation.

    The project has evolved through multiple Grad-CAM
    implementations, so this function inspects the function
    signature and passes the arguments it understands.
    """

    function = find_gradcam_function()

    if function is None:
        raise RuntimeError(
            "Grad-CAM function could not be imported."
        )

    signature = inspect.signature(function)

    parameters = signature.parameters

    kwargs = {}

    # Image parameter
    for name in [
        "image_path",
        "img_path",
        "file_path",
        "path",
        "image",
    ]:

        if name in parameters:

            kwargs[name] = str(
                image_path
            )

            break

    # Grade / target parameter
    for name in [
        "predicted_grade",
        "target_grade",
        "grade",
        "target_class",
        "class_idx",
    ]:

        if name in parameters:

            kwargs[name] = predicted_grade

            break

    # Output directory if supported
    output_dir = (
        Path("artifacts")
        / "gradcam"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in [
        "output_dir",
        "save_dir",
        "out_dir",
    ]:

        if name in parameters:

            kwargs[name] = str(
                output_dir
            )

            break

    # --------------------------------------------------------
    # Call using keyword arguments if possible
    # --------------------------------------------------------

    if kwargs:

        try:
            return function(**kwargs)

        except TypeError:
            pass

    # --------------------------------------------------------
    # Fallback: image path as first positional argument
    # --------------------------------------------------------

    return function(
        str(image_path)
    )


def find_image_files(value):
    """
    Recursively search a Grad-CAM result for generated images.
    """

    found = []

    if value is None:
        return found

    if isinstance(
        value,
        (str, Path),
    ):

        path = Path(value)

        if path.exists() and path.is_file():

            if path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }:

                found.append(path)

        return found

    if isinstance(value, dict):

        for item in value.values():

            found.extend(
                find_image_files(item)
            )

        return found

    if isinstance(value, (list, tuple)):

        for item in value:

            found.extend(
                find_image_files(item)
            )

        return found

    return found


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🦴 Knee AI"
    )

    st.caption(
        "Osteoarthritis Detection"
    )

    st.divider()

    st.markdown(
        "### Navigation"
    )

    page = st.radio(
        "Navigation",
        [
            "Prediction",
            "Model Performance",
            "Explainability",
            "System",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        "### Model"
    )

    st.caption(
        f"Architecture: {MODEL_ARCHITECTURE}"
    )

    st.caption(
        f"Experiment: {MODEL_EXPERIMENT}"
    )

    st.caption(
        f"Input: {IMAGE_SIZE} × {IMAGE_SIZE}"
    )

    st.caption(
        "Output: 5 severity grades"
    )

    st.divider()

    st.markdown(
        """
        <div class="status-box">
            ● Model service available
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "Research / educational prototype. "
        "Not intended for independent medical "
        "diagnosis or treatment decisions."
    )


# ============================================================
# HERO
# ============================================================

st.title(
    "AI Knee Osteoarthritis Detection"
)

st.write(
    """
    Automated knee X-ray severity assessment using an
    **Ordinal ResNet50** deep-learning model with
    probability estimation and Grad-CAM explainability.
    """
)

st.divider()


# ============================================================
# KPI ROW
# ============================================================

accuracy = PERFORMANCE.get(
    "accuracy",
    0.6093,
)

macro_f1 = PERFORMANCE.get(
    "macro_f1",
    0.5989,
)

qwk = PERFORMANCE.get(
    "quadratic_weighted_kappa",
    0.7707,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.metric(
        "Accuracy",
        f"{float(accuracy) * 100:.2f}%",
    )

with kpi2:

    st.metric(
        "Macro F1",
        f"{float(macro_f1) * 100:.2f}%",
    )

with kpi3:

    st.metric(
        "Quadratic Weighted Kappa",
        f"{float(qwk) * 100:.2f}%",
    )

with kpi4:

    st.metric(
        "Severity Grades",
        "5",
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "Prediction":

    st.header(
        "Knee X-ray Analysis"
    )

    st.write(
        """
        Upload a knee X-ray image to obtain an automated
        osteoarthritis severity prediction.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload knee X-ray",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        help=(
            "Supported formats: PNG, JPG and JPEG."
        ),
    )

    if uploaded_file is None:

        st.info(
            "Upload a knee X-ray image to begin analysis."
        )

        st.subheader(
            "How the system works"
        )

        flow1, flow2, flow3, flow4 = st.columns(4)

        with flow1:
            st.markdown(
                "### 01\nUpload"
            )
            st.caption(
                "Provide a knee X-ray image."
            )

        with flow2:
            st.markdown(
                "### 02\nPreprocess"
            )
            st.caption(
                "Resize and normalize the image."
            )

        with flow3:
            st.markdown(
                "### 03\nPredict"
            )
            st.caption(
                "Ordinal ResNet50 predicts severity."
            )

        with flow4:
            st.markdown(
                "### 04\nExplain"
            )
            st.caption(
                "Grad-CAM highlights model attention."
            )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_col, analysis_col = st.columns(
            [1, 1],
            gap="large",
        )

        with image_col:

            st.subheader(
                "Uploaded X-ray"
            )

            st.image(
                image,
                width="stretch",
            )

            st.caption(
                f"{uploaded_file.name} · "
                f"{image.width} × {image.height}px"
            )

        with analysis_col:

            st.subheader(
                "Run Analysis"
            )

            st.write(
                """
                The uploaded image will be passed through
                the same inference pipeline used by the
                deployed application.
                """
            )

            analyze = st.button(
                "Run AI Prediction",
                width="stretch",
                type="primary",
            )

            if analyze:

                image_path = (
                    save_uploaded_file(
                        uploaded_file
                    )
                )

                with st.spinner(
                    "Running Ordinal ResNet50 inference..."
                ):

                    try:

                        result = predict(
                            image_path
                        )

                        st.session_state[
                            "prediction_result"
                        ] = result

                        st.session_state[
                            "prediction_image"
                        ] = str(
                            image_path
                        )

                    except Exception as exc:

                        st.error(
                            "Prediction failed."
                        )

                        st.exception(
                            exc
                        )

        result = st.session_state.get(
            "prediction_result"
        )

        if result is not None:

            st.divider()

            predicted_grade = extract_grade(
                result
            )

            confidence = extract_confidence(
                result
            )

            st.header(
                "Prediction Result"
            )

            result_col, probability_col = st.columns(
                [0.8, 1.2],
                gap="large",
            )

            with result_col:

                st.success(
                    f"Predicted severity: "
                    f"**Grade {predicted_grade}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%",
                )

                st.markdown(
                    f"**Clinical severity category**"
                )

                st.write(
                    GRADE_DESCRIPTIONS.get(
                        predicted_grade,
                        "Severity classification",
                    )
                )

            with probability_col:

                st.subheader(
                    "Class Probability Distribution"
                )

                probabilities = (
                    extract_probabilities(
                        result
                    )
                )

                if probabilities is not None:

                    for grade, probability in zip(
                        GRADE_NAMES,
                        probabilities,
                    ):

                        label_col, value_col = st.columns(
                            [4, 1]
                        )

                        with label_col:

                            st.write(
                                grade
                            )

                        with value_col:

                            st.write(
                                f"{probability:.2f}%"
                            )

                        st.progress(
                            min(
                                max(
                                    probability / 100,
                                    0,
                                ),
                                1,
                            )
                        )

                else:

                    st.warning(
                        "Class probability information "
                        "was not returned by the prediction pipeline."
                    )

            # ------------------------------------------------
            # RAW RESULT
            # ------------------------------------------------

            with st.expander(
                "View prediction details"
            ):

                try:

                    st.json(
                        normalize_result_for_json(
                            result
                        )
                    )

                except Exception:

                    st.write(
                        result
                    )

            # ------------------------------------------------
            # QUICK NEXT STEP
            # ------------------------------------------------

            st.info(
                "For an explanation of the model's visual "
                "attention, open the **Explainability** section."
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "Model Performance"
    )

    st.write(
        """
        Performance reported for the deployed
        **Experiment 5 Ordinal ResNet50** model.
        """
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "Accuracy",
            f"{float(accuracy) * 100:.2f}%",
        )

    with p2:

        st.metric(
            "Macro F1",
            f"{float(macro_f1) * 100:.2f}%",
        )

    with p3:

        st.metric(
            "Quadratic Weighted Kappa",
            f"{float(qwk) * 100:.2f}%",
        )

    st.divider()

    st.subheader(
        "Evaluation Metrics"
    )

    performance_table = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro F1",
                "Quadratic Weighted Kappa",
            ],
            "Score": [
                float(accuracy),
                float(macro_f1),
                float(qwk),
            ],
        }
    )

    performance_table[
        "Score"
    ] = (
        performance_table[
            "Score"
        ] * 100
    ).round(2)

    performance_table[
        "Interpretation"
    ] = [
        "Overall classification accuracy",
        "Average F1 performance across grades",
        "Ordinal agreement metric",
    ]

    st.dataframe(
        performance_table,
        hide_index=True,
        width="stretch",
    )

    st.divider()

    st.subheader(
        "Severity Classes"
    )

    classes = pd.DataFrame(
        {
            "Grade": [
                "Grade 0",
                "Grade 1",
                "Grade 2",
                "Grade 3",
                "Grade 4",
            ],
            "Description": [
                GRADE_DESCRIPTIONS[i]
                for i in range(5)
            ],
        }
    )

    st.dataframe(
        classes,
        hide_index=True,
        width="stretch",
    )

    st.divider()

    st.subheader(
        "Model Configuration"
    )

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.write(
            "**Architecture**"
        )

        st.code(
            MODEL_ARCHITECTURE
        )

        st.write(
            "**Experiment**"
        )

        st.code(
            MODEL_EXPERIMENT
        )

        st.write(
            "**Input size**"
        )

        st.code(
            f"{IMAGE_SIZE} × {IMAGE_SIZE} × 3"
        )

    with config_col2:

        st.write(
            "**Number of classes**"
        )

        st.code(
            "5"
        )

        st.write(
            "**Ordinal thresholds**"
        )

        st.code(
            "4"
        )

        st.write(
            "**Inference device**"
        )

        st.code(
            "CPU / auto"
        )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.header(
        "Explainable AI"
    )

    st.write(
        """
        Grad-CAM provides a visual explanation of which
        regions of the X-ray contributed to the model's
        prediction.
        """
    )

    image_path = st.session_state.get(
        "prediction_image"
    )

    result = st.session_state.get(
        "prediction_result"
    )

    if image_path is None or result is None:

        st.info(
            "Run a prediction first. "
            "The Grad-CAM explanation will then use "
            "the same uploaded X-ray."
        )

    else:

        predicted_grade = extract_grade(
            result
        )

        st.success(
            f"Current prediction: "
            f"Grade {predicted_grade}"
        )

        generate = st.button(
            "Generate Grad-CAM",
            width="stretch",
            type="primary",
        )

        if generate:

            with st.spinner(
                "Generating Grad-CAM visualization..."
            ):

                try:

                    gradcam_result = run_gradcam(
                        image_path,
                        predicted_grade,
                    )

                    st.session_state[
                        "gradcam_result"
                    ] = gradcam_result

                    st.success(
                        "Grad-CAM generated successfully."
                    )

                except Exception as exc:

                    st.error(
                        "Grad-CAM generation failed."
                    )

                    st.exception(
                        exc
                    )

        gradcam_result = st.session_state.get(
            "gradcam_result"
        )

        if gradcam_result is not None:

            st.divider()

            st.subheader(
                "Generated Explanation"
            )

            # --------------------------------------------
            # Handle dictionary outputs
            # --------------------------------------------

            if isinstance(
                gradcam_result,
                dict,
            ):

                preferred_items = []

                for key, value in gradcam_result.items():

                    if isinstance(
                        value,
                        (str, Path),
                    ):

                        path = Path(value)

                        if (
                            path.exists()
                            and path.is_file()
                            and path.suffix.lower()
                            in {
                                ".jpg",
                                ".jpeg",
                                ".png",
                                ".webp",
                            }
                        ):

                            preferred_items.append(
                                (
                                    key,
                                    path,
                                )
                            )

                if preferred_items:

                    cols = st.columns(
                        min(
                            len(
                                preferred_items
                            ),
                            3,
                        )
                    )

                    for index, (
                        key,
                        path,
                    ) in enumerate(
                        preferred_items
                    ):

                        with cols[
                            index % len(cols)
                        ]:

                            st.caption(
                                str(key).replace(
                                    "_",
                                    " ",
                                ).title()
                            )

                            st.image(
                                str(path),
                                width="stretch",
                            )

                else:

                    image_files = (
                        find_image_files(
                            gradcam_result
                        )
                    )

                    if image_files:

                        cols = st.columns(
                            min(
                                len(
                                    image_files
                                ),
                                3,
                            )
                        )

                        for index, path in enumerate(
                            image_files
                        ):

                            with cols[
                                index % len(cols)
                            ]:

                                st.caption(
                                    path.name
                                )

                                st.image(
                                    str(path),
                                    width="stretch",
                                )

                    else:

                        st.json(
                            normalize_result_for_json(
                                gradcam_result
                            )
                        )

            else:

                image_files = (
                    find_image_files(
                        gradcam_result
                    )
                )

                if image_files:

                    cols = st.columns(
                        min(
                            len(image_files),
                            3,
                        )
                    )

                    for index, path in enumerate(
                        image_files
                    ):

                        with cols[
                            index % len(cols)
                        ]:

                            st.caption(
                                path.name
                            )

                            st.image(
                                str(path),
                                width="stretch",
                            )

                else:

                    st.write(
                        gradcam_result
                    )

        st.divider()

        st.subheader(
            "Why Grad-CAM?"
        )

        st.write(
            """
            The prediction probability alone does not show
            where the model is focusing. Grad-CAM adds an
            interpretability layer by producing activation
            maps over the input image.

            This is particularly useful for a medical imaging
            research prototype because it allows the prediction
            to be visually inspected rather than treated as
            an unexplained classification score.
            """
        )


# ============================================================
# SYSTEM
# ============================================================

elif page == "System":

    st.header(
        "System Architecture"
    )

    st.write(
        """
        The project is organized as a deployable machine
        learning application rather than a standalone notebook.
        """
    )

    st.divider()

    architecture = [
        (
            "01",
            "Streamlit",
            "Interactive X-ray analysis interface",
        ),
        (
            "02",
            "FastAPI",
            "REST inference API",
        ),
        (
            "03",
            "Ordinal ResNet50",
            "Five-grade severity classification",
        ),
        (
            "04",
            "Hugging Face Hub",
            "Remote model artifact storage",
        ),
        (
            "05",
            "Docker",
            "Containerized deployment",
        ),
        (
            "06",
            "Pytest",
            "Automated testing",
        ),
        (
            "07",
            "Monitoring",
            "Prediction and system monitoring",
        ),
        (
            "08",
            "Grad-CAM",
            "Visual explainability",
        ),
    ]

    for row_start in range(
        0,
        len(architecture),
        4,
    ):

        row = architecture[
            row_start:row_start + 4
        ]

        cols = st.columns(
            len(row)
        )

        for col, (
            number,
            title,
            description,
        ) in zip(
            cols,
            row,
        ):

            with col:

                st.markdown(
                    f"### {number}"
                )

                st.markdown(
                    f"**{title}**"
                )

                st.caption(
                    description
                )

    st.divider()

    st.subheader(
        "Deployment Information"
    )

    deploy1, deploy2 = st.columns(2)

    with deploy1:

        st.write(
            "**Model repository**"
        )

        st.code(
            MODEL_REPOSITORY
        )

        st.write(
            "**Architecture**"
        )

        st.code(
            MODEL_ARCHITECTURE
        )

        st.write(
            "**Model experiment**"
        )

        st.code(
            MODEL_EXPERIMENT
        )

    with deploy2:

        st.write(
            "**Model SHA256**"
        )

        st.code(
            MODEL_SHA256
        )

        st.write(
            "**Input**"
        )

        st.code(
            f"{IMAGE_SIZE} × {IMAGE_SIZE} RGB"
        )

        st.write(
            "**Output**"
        )

        st.code(
            "Grade 0 → Grade 4"
        )

    st.divider()

    st.subheader(
        "Engineering Stack"
    )

    technologies = [
        "Python",
        "PyTorch",
        "Torchvision",
        "ResNet50",
        "Ordinal Classification",
        "OpenCV",
        "Pillow",
        "Albumentations",
        "FastAPI",
        "Streamlit",
        "Grad-CAM",
        "Docker",
        "Docker Compose",
        "Pytest",
        "GitHub Actions",
        "Hugging Face",
        "MLflow",
        "Prometheus",
    ]

    st.write(
        " · ".join(
            technologies
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-line">

        <strong>AI Knee Osteoarthritis Detection</strong>
        <br>

        Ordinal ResNet50 · Experiment 5 · Explainable AI

        <br><br>

        Research / educational prototype.
        Model predictions should not be used as an
        independent medical diagnosis.

    </div>
    """,
    unsafe_allow_html=True,
)
