from pathlib import Path
from typing import Any
import inspect
import tempfile
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
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
# CONSTANTS
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

def get_class_probabilities(result):
    """
    Extract Grade 0-4 probabilities from predict() output.
    Returns percentages.
    """

    grade_probabilities = result.get(
        "grade_probabilities",
        {}
    )

    probabilities = []

    for grade in ["0", "1", "2", "3", "4"]:

        values = grade_probabilities.get(
            grade,
            {}
        )

        probability = values.get(
            "probability_percent",
            0.0
        )

        probabilities.append(
            float(probability)
        )

    return probabilities
    
# ============================================================
# MODEL CONFIG
# ============================================================

try:
    MODEL_CONFIG = get_model_config()
except Exception:
    MODEL_CONFIG = {}

if not isinstance(MODEL_CONFIG, dict):
    MODEL_CONFIG = {}

MODEL_INFO = MODEL_CONFIG.get("model", {})
PERFORMANCE = MODEL_CONFIG.get("performance", {})

if not isinstance(MODEL_INFO, dict):
    MODEL_INFO = {}

if not isinstance(PERFORMANCE, dict):
    PERFORMANCE = {}

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
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "prediction_result": None,
    "prediction_image": None,
    "prediction_filename": None,
    "gradcam_result": None,
    "gradcam_images": [],
    "analysis_complete": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

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

h1, h2, h3, h4 {
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


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0b0d13 0%,
            #090a0e 100%
        );

    border-right:
        1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] * {
    color: #d7d9e0;
}


/* =========================================================
   METRICS
   ========================================================= */

div[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(99,102,241,0.09),
            rgba(255,255,255,0.02)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 18px;
    padding: 20px;
    min-height: 125px;

    box-shadow:
        0 12px 30px rgba(0,0,0,0.12);
}

div[data-testid="stMetricLabel"] {
    color: #858a98 !important;
    font-size: 0.75rem !important;
}

div[data-testid="stMetricValue"] {
    color: #f4f4f5 !important;
    font-weight: 800 !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    border-radius: 11px;
    min-height: 44px;

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.18),
            rgba(59,130,246,0.10)
        );

    border:
        1px solid rgba(129,140,248,0.25);

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
        rgba(129,140,248,0.55);

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.28),
            rgba(59,130,246,0.17)
        );

    color: white;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background:
        rgba(255,255,255,0.018);

    border:
        1px dashed rgba(255,255,255,0.13);

    border-radius: 18px;
    padding: 8px;
}


/* =========================================================
   TABS / RADIO
   ========================================================= */

button[data-baseweb="tab"] {
    font-weight: 650;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #a5b4fc;
}


/* =========================================================
   ALERTS
   ========================================================= */

div[data-testid="stAlert"] {
    border-radius: 12px;
}


/* =========================================================
   EXPANDERS
   ========================================================= */

details {
    border-radius: 14px !important;
    border-color:
        rgba(255,255,255,0.08) !important;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {
    padding: 24px;
    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.035),
            rgba(255,255,255,0.012)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    margin-bottom: 18px;
}

.grade-card {
    padding: 22px;
    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(16,185,129,0.10),
            rgba(255,255,255,0.02)
        );

    border:
        1px solid rgba(52,211,153,0.20);
}

.probability-row {
    padding: 10px 0;
}


/* =========================================================
   STATUS
   ========================================================= */

.status-box {
    padding: 12px 15px;
    border-radius: 12px;

    background:
        rgba(52,211,153,0.06);

    border:
        1px solid rgba(52,211,153,0.14);

    color: #a7f3d0;
    font-size: 0.82rem;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer-line {
    margin-top: 60px;
    padding-top: 25px;

    border-top:
        1px solid rgba(255,255,255,0.07);

    text-align: center;

    color: #626775;

    font-size: 0.75rem;
    line-height: 1.7;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 900px) {

    .block-container {
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# GENERIC HELPERS
# ============================================================

def get_value(result: Any, *keys, default=None):
    """
    Safely retrieve a value from dictionaries or objects.
    """

    if result is None:
        return default

    if isinstance(result, dict):

        for key in keys:
            if key in result:
                return result[key]

        return default

    for key in keys:

        if hasattr(result, key):
            return getattr(result, key)

    return default


def recursive_find(result, wanted_keys):
    """
    Recursively search nested dictionaries/lists/objects.
    """

    if result is None:
        return None

    wanted_keys = {
        str(k).lower()
        for k in wanted_keys
    }

    if isinstance(result, dict):

        for key, value in result.items():

            if str(key).lower() in wanted_keys:
                return value

        for value in result.values():

            found = recursive_find(
                value,
                wanted_keys,
            )

            if found is not None:
                return found

    elif isinstance(result, (list, tuple)):

        for value in result:

            found = recursive_find(
                value,
                wanted_keys,
            )

            if found is not None:
                return found

    return None


def to_percentage(value):
    """
    Convert 0.85 -> 85
    Convert 85 -> 85
    """

    if value is None:
        return 0.0

    try:
        value = float(value)
    except Exception:
        return 0.0

    if not math.isfinite(value):
        return 0.0

    if 0 <= value <= 1:
        return value * 100

    return value


def softmax(values):
    """
    Convert logits/scores into probabilities.
    """

    arr = np.asarray(
        values,
        dtype=float,
    )

    if arr.ndim != 1:
        arr = arr.flatten()

    if len(arr) == 0:
        return None

    if not np.all(np.isfinite(arr)):
        return None

    arr = arr - np.max(arr)

    exp_values = np.exp(arr)

    total = np.sum(exp_values)

    if total <= 0:
        return None

    return (
        exp_values / total
    ).tolist()


# ============================================================
# PREDICTION EXTRACTION
# ============================================================

def extract_grade(result):

    value = recursive_find(
        result,
        {
            "predicted_grade",
            "prediction",
            "grade",
            "predicted_class",
            "class_id",
            "class_idx",
        },
    )

    if value is None:
        return None

    try:

        if isinstance(value, str):

            text = value.lower().strip()

            if "grade" in text:

                text = (
                    text
                    .replace("grade", "")
                    .strip()
                )

            value = float(text)

        return int(value)

    except Exception:
        return None


def extract_confidence(result):

    value = recursive_find(
        result,
        {
            "confidence",
            "prediction_confidence",
            "max_probability",
            "max_prob",
        },
    )

    if value is not None:
        return to_percentage(value)

    probabilities = extract_probabilities(
        result,
        allow_logits=True,
    )

    if probabilities:

        return max(probabilities)

    return None


# ============================================================
# PROBABILITY EXTRACTION
# ============================================================

def extract_probabilities(
    result,
    allow_logits=True,
):
    """
    Robustly extract Grade 0-4 probabilities.

    Supports:
        probabilities
        class_probabilities
        grade_probabilities
        probs
        probability_distribution
        logits
        class_logits
        scores

    Nested result dictionaries are also supported.
    """

    probability_keys = {
        "class_probabilities",
        "grade_probabilities",
        "probabilities",
        "probs",
        "probability_distribution",
        "class_probability",
    }

    values = recursive_find(
        result,
        probability_keys,
    )

    if values is not None:

        parsed = parse_probability_values(
            values
        )

        if parsed is not None:
            return parsed

    if allow_logits:

        logits = recursive_find(
            result,
            {
                "logits",
                "class_logits",
                "raw_logits",
                "scores",
            },
        )

        if logits is not None:

            try:

                if isinstance(logits, dict):

                    ordered = []

                    for i in range(5):

                        value = (
                            logits.get(
                                f"Grade {i}"
                            )

                            if f"Grade {i}" in logits

                            else logits.get(
                                str(i)
                            )
                        )

                        if value is None:
                            return None

                        ordered.append(
                            float(value)
                        )

                    logits = ordered

                probabilities = softmax(
                    logits
                )

                if (
                    probabilities
                    and len(probabilities) == 5
                ):
                    return [
                        float(x) * 100
                        for x in probabilities
                    ]

            except Exception:
                pass

    return None


def parse_probability_values(values):

    if values is None:
        return None

    if isinstance(values, dict):

        result = []

        for i in range(5):

            candidates = [
                f"Grade {i}",
                f"grade_{i}",
                f"grade{i}",
                str(i),
                i,
            ]

            found = None

            for key in candidates:

                if key in values:
                    found = values[key]
                    break

            if found is None:
                return None

            result.append(
                to_percentage(found)
            )

        return normalize_probabilities(
            result
        )

    try:

        arr = list(values)

        if len(arr) != 5:
            return None

        parsed = [
            to_percentage(v)
            for v in arr
        ]

        return normalize_probabilities(
            parsed
        )

    except Exception:
        return None


def normalize_probabilities(probabilities):

    if probabilities is None:
        return None

    arr = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(arr) != 5:
        return None

    if not np.all(np.isfinite(arr)):
        return None

    arr = np.maximum(
        arr,
        0,
    )

    total = arr.sum()

    if total <= 0:
        return None

    # If values were percentages, normalize.
    arr = (
        arr / total
    ) * 100

    return arr.tolist()


# ============================================================
# FILE HANDLING
# ============================================================

def save_uploaded_file(
    uploaded_file,
):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:

        tmp.write(
            uploaded_file.getbuffer()
        )

        return Path(
            tmp.name
        )


# ============================================================
# JSON-SAFE RESULT
# ============================================================

def normalize_result_for_json(
    result,
):

    try:

        if isinstance(
            result,
            dict,
        ):
            return result

        if hasattr(
            result,
            "__dict__",
        ):
            return result.__dict__

        return str(result)

    except Exception:

        return str(result)


# ============================================================
# GRAD-CAM
# ============================================================

def find_gradcam_function():

    try:

        from src.explainability.gradcam import (
            generate_gradcam,
        )

        return generate_gradcam

    except Exception as exc:

        return None


def run_gradcam(
    image_path,
    predicted_grade,
):

    function = find_gradcam_function()

    if function is None:

        raise RuntimeError(
            "Could not import "
            "src.explainability.gradcam.generate_gradcam"
        )

    output_dir = (
        Path("artifacts")
        / "gradcam"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    signature = inspect.signature(
        function
    )

    parameters = signature.parameters

    kwargs = {}

    # ------------------------------------------
    # IMAGE ARGUMENT
    # ------------------------------------------

    image_names = [
        "image_path",
        "img_path",
        "file_path",
        "path",
        "image",
        "input_path",
    ]

    for name in image_names:

        if name in parameters:

            kwargs[name] = str(
                image_path
            )

            break

    # ------------------------------------------
    # TARGET / GRADE
    # ------------------------------------------

    grade_names = [
        "predicted_grade",
        "target_grade",
        "grade",
        "target_class",
        "class_idx",
        "target",
    ]

    for name in grade_names:

        if name in parameters:

            kwargs[name] = int(
                predicted_grade
            )

            break

    # ------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------

    for name in [
        "output_dir",
        "save_dir",
        "out_dir",
        "output_path",
    ]:

        if name in parameters:

            kwargs[name] = str(
                output_dir
            )

            break

    # ------------------------------------------
    # CALL
    # ------------------------------------------

    try:

        if kwargs:

            return function(
                **kwargs
            )

    except TypeError:

        pass

    # Fallback
    try:

        return function(
            str(image_path)
        )

    except TypeError:

        return function(
            str(image_path),
            int(predicted_grade),
        )


def collect_gradcam_files(
    value=None,
):

    files = []

    # ------------------------------------------
    # VALUE RETURNED BY FUNCTION
    # ------------------------------------------

    if value is not None:

        files.extend(
            recursive_image_search(
                value
            )
        )

    # ------------------------------------------
    # SEARCH STANDARD OUTPUT DIRECTORY
    # ------------------------------------------

    search_dirs = [
        Path("artifacts/gradcam"),
        Path("artifacts"),
        Path("gradcam"),
        Path("outputs"),
    ]

    for directory in search_dirs:

        if not directory.exists():
            continue

        try:

            for path in directory.rglob("*"):

                if (
                    path.is_file()
                    and path.suffix.lower()
                    in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                    }
                ):

                    files.append(
                        path
                    )

        except Exception:
            continue

    # ------------------------------------------
    # DEDUPLICATE
    # ------------------------------------------

    unique = []

    seen = set()

    for path in files:

        try:

            key = str(
                Path(path).resolve()
            )

        except Exception:

            key = str(path)

        if key not in seen:

            seen.add(key)

            unique.append(
                Path(path)
            )

    return unique


def recursive_image_search(
    value,
):

    found = []

    if value is None:
        return found

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
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }
        ):

            found.append(
                path
            )

        return found

    if isinstance(
        value,
        dict,
    ):

        for item in value.values():

            found.extend(
                recursive_image_search(
                    item
                )
            )

        return found

    if isinstance(
        value,
        (list, tuple),
    ):

        for item in value:

            found.extend(
                recursive_image_search(
                    item
                )
            )

        return found

    return found


# ============================================================
# DISPLAY GRAD-CAM
# ============================================================

def display_gradcam_images(
    image_paths,
):

    if not image_paths:

        st.warning(
            "Grad-CAM completed, but no image file "
            "was returned or found in the output folders."
        )

        return

    # Keep only existing images
    image_paths = [
        Path(p)
        for p in image_paths
        if Path(p).exists()
    ]

    if not image_paths:
        return

    # Remove duplicate filenames
    unique = []

    seen = set()

    for path in image_paths:

        key = str(
            path.resolve()
        )

        if key not in seen:

            seen.add(key)
            unique.append(path)

    image_paths = unique

    # ------------------------------------------
    # Identify likely image types
    # ------------------------------------------

    original = []
    heatmaps = []
    overlays = []
    others = []

    for path in image_paths:

        name = path.stem.lower()

        if (
            "original" in name
            or "input" in name
            or "source" in name
        ):
            original.append(path)

        elif (
            "overlay" in name
            or "cam_overlay" in name
            or "heatmap_overlay" in name
        ):
            overlays.append(path)

        elif (
            "heatmap" in name
            or "gradcam" in name
            or "cam" in name
        ):
            heatmaps.append(path)

        else:
            others.append(path)

    ordered = (
        original
        + heatmaps
        + overlays
        + others
    )

    # ------------------------------------------
    # Display maximum 4
    # ------------------------------------------

    ordered = ordered[:4]

    cols = st.columns(
        min(
            len(ordered),
            3,
        )
    )

    for index, path in enumerate(
        ordered
    ):

        with cols[
            index % len(cols)
        ]:

            name = path.stem.replace(
                "_",
                " ",
            ).title()

            st.markdown(
                f"**{name}**"
            )

            try:

                st.image(
                    str(path),
                    width="stretch",
                )

            except Exception as exc:

                st.error(
                    f"Could not display {path.name}"
                )

                st.caption(
                    str(exc)
                )


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
# GLOBAL KPI ROW
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
# PREDICTION
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
            "webp",
        ],
        help="Supported formats: PNG, JPG, JPEG and WEBP.",
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

        image_col, control_col = st.columns(
            [1.15, 0.85],
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

        with control_col:

            st.subheader(
                "AI Analysis"
            )

            st.write(
                """
                The image will be passed through the
                deployed inference pipeline.
                """
            )

            analyze = st.button(
                "Run AI Prediction",
                width="stretch",
                type="primary",
            )

            if analyze:

                # Clear old analysis
                st.session_state[
                    "prediction_result"
                ] = None

                st.session_state[
                    "prediction_image"
                ] = None

                st.session_state[
                    "prediction_filename"
                ] = None

                st.session_state[
                    "gradcam_result"
                ] = None

                st.session_state[
                    "gradcam_images"
                ] = []

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

                        st.session_state[
                            "prediction_filename"
                        ] = uploaded_file.name

                        st.session_state[
                            "analysis_complete"
                        ] = True

                        st.success(
                            "Prediction completed successfully."
                        )

                    except Exception as exc:

                        st.session_state[
                            "analysis_complete"
                        ] = False

                        st.error(
                            "Prediction failed."
                        )

                        st.exception(
                            exc
                        )

        # ====================================================
        # RESULT
        # ====================================================

        result = st.session_state.get(
            "prediction_result"
        )

        if result is not None:

            st.divider()

            predicted_grade = (
                extract_grade(
                    result
                )
            )

            confidence = (
                extract_confidence(
                    result
                )
            )

            probabilities = (
                extract_probabilities(
                    result
                )
            )

            st.header(
                "Prediction Result"
            )

            # -----------------------------------------------
            # RESULT HEADER
            # -----------------------------------------------

            result_col, confidence_col = st.columns(
                [1, 1],
                gap="large",
            )

            with result_col:

                if predicted_grade is not None:

                    st.markdown(
                        f"""
                        <div class="grade-card">

                        <h3>Predicted Severity</h3>

                        <h1>
                        Grade {predicted_grade}
                        </h1>

                        <p>
                        {
                            GRADE_DESCRIPTIONS.get(
                                predicted_grade,
                                "Severity classification"
                            )
                        }
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.warning(
                        "Prediction grade was not returned "
                        "in the expected format."
                    )

            with confidence_col:

                if confidence is not None:

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%",
                    )

                else:

                    st.metric(
                        "Prediction Confidence",
                        "Not available",
                    )

            # ============================================================
            # CLASS PROBABILITY DISTRIBUTION
            # ============================================================

            st.divider()

            st.subheader(
             "Class Probability Distribution"
            )

            probabilities = get_class_probabilities(
            result
            )

            probability_df = pd.DataFrame({

            "Grade": [
            "Grade 0",
            "Grade 1",
            "Grade 2",
            "Grade 3",
            "Grade 4"
            ],

            "Probability": probabilities
            })


            # ============================================================
            # PROBABILITY CHART
            # ============================================================

            fig = px.bar(

            probability_df,

            x="Grade",

            y="Probability",

            text="Probability",

            labels={
        "Probability": "Probability (%)",
        "Grade": "Severity Grade"
            },

            title="Model Class Probability Distribution"

            )

            fig.update_traces(

            texttemplate="%{text:.2f}%",

            textposition="outside"

            )

            fig.update_layout(

            yaxis=dict(

            range=[
            0,
            max(
                100,
                max(probabilities) + 10
            )
            ],

            ticksuffix="%"

            ),

            xaxis_title="Severity Grade",

            yaxis_title="Probability (%)",

            height=450,

            showlegend=False

            )

            st.plotly_chart(

            fig,

            use_container_width=True

            )

            # ============================================================
            # PROBABILITY METRICS
            # ============================================================

            cols = st.columns(5)

            for i, probability in enumerate(probabilities):

                with cols[i]:

                     st.metric(

            f"Grade {i}",

            f"{probability:.2f}%"

             )
                    
            # -----------------------------------------------
            # CLINICAL INTERPRETATION
            # -----------------------------------------------

            if predicted_grade is not None:

                st.divider()

                st.subheader(
                    "Severity Interpretation"
                )

                interpretation = pd.DataFrame(
                    {
                        "Grade": [
                            GRADE_NAMES[predicted_grade]
                        ],
                        "Clinical Category": [
                            GRADE_DESCRIPTIONS.get(
                                predicted_grade
                            )
                        ],
                    }
                )

                st.dataframe(
                    interpretation,
                    hide_index=True,
                    width="stretch",
                )

            # -----------------------------------------------
            # RAW RESULT
            # -----------------------------------------------

            with st.expander(
                "View raw prediction response"
            ):

                st.json(
                    normalize_result_for_json(
                        result
                    )
                )

            # -----------------------------------------------
            # EXPLAINABILITY BUTTON
            # -----------------------------------------------

            st.info(
                "Prediction completed. Open "
                "**Explainability** in the sidebar to generate "
                "the Grad-CAM visualization."
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "Model Performance"
    )

    st.write(
        f"""
        Evaluation results for the deployed
        **{MODEL_EXPERIMENT} {MODEL_ARCHITECTURE}**
        model.
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
                float(accuracy) * 100,
                float(macro_f1) * 100,
                float(qwk) * 100,
            ],
            "Interpretation": [
                "Overall classification accuracy",
                "Average F1 performance across grades",
                "Ordinal agreement metric",
            ],
        }
    )

    performance_table[
        "Score"
    ] = performance_table[
        "Score"
    ].round(2)

    performance_table[
        "Score"
    ] = performance_table[
        "Score"
    ].astype(str) + "%"

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
            "Grade": GRADE_NAMES,
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

    c1, c2 = st.columns(2)

    with c1:

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
            "**Input Size**"
        )

        st.code(
            f"{IMAGE_SIZE} × {IMAGE_SIZE} × 3"
        )

    with c2:

        st.write(
            "**Number of Classes**"
        )

        st.code(
            "5"
        )

        st.write(
            "**Ordinal Thresholds**"
        )

        st.code(
            "4"
        )

        st.write(
            "**Inference Device**"
        )

        st.code(
            "CPU / Auto"
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
        Grad-CAM provides a visual explanation of the
        image regions contributing to the model prediction.
        """
    )

    image_path = st.session_state.get(
        "prediction_image"
    )

    result = st.session_state.get(
        "prediction_result"
    )

    if (
        image_path is None
        or result is None
    ):

        st.info(
            "Run a prediction first. "
            "The same uploaded X-ray will then be "
            "available for Grad-CAM analysis."
        )

    else:

        predicted_grade = (
            extract_grade(
                result
            )
        )

        if predicted_grade is None:

            st.warning(
                "Unable to determine the predicted grade "
                "from the prediction response."
            )

        else:

            st.success(
                f"Current prediction: Grade {predicted_grade}"
            )

            # -----------------------------------------------
            # SOURCE IMAGE
            # -----------------------------------------------

            st.subheader(
                "Input X-ray"
            )

            try:

                st.image(
                    image_path,
                    width="stretch",
                )

            except Exception:

                st.warning(
                    "Original X-ray could not be displayed."
                )

            # -----------------------------------------------
            # GENERATE
            # -----------------------------------------------

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

                        gradcam_result = (
                            run_gradcam(
                                image_path,
                                predicted_grade,
                            )
                        )

                        image_files = (
                            collect_gradcam_files(
                                gradcam_result
                            )
                        )

                        st.session_state[
                            "gradcam_result"
                        ] = gradcam_result

                        st.session_state[
                            "gradcam_images"
                        ] = [
                            str(path)
                            for path
                            in image_files
                        ]

                        if image_files:

                            st.success(
                                "Grad-CAM generated successfully."
                            )

                        else:

                            st.warning(
                                "Grad-CAM function completed, "
                                "but no output image was found."
                            )

                    except Exception as exc:

                        st.session_state[
                            "gradcam_result"
                        ] = None

                        st.session_state[
                            "gradcam_images"
                        ] = []

                        st.error(
                            "Grad-CAM generation failed."
                        )

                        st.exception(
                            exc
                        )

            # -----------------------------------------------
            # DISPLAY
            # -----------------------------------------------

            saved_images = (
                st.session_state.get(
                    "gradcam_images",
                    [],
                )
            )

            if saved_images:

                st.divider()

                st.subheader(
                    "Generated Explanation"
                )

                display_gradcam_images(
                    saved_images
                )

            else:

                if st.session_state.get(
                    "gradcam_result"
                ) is not None:

                    st.warning(
                        "Grad-CAM returned a result, but "
                        "no displayable image was found."
                    )

                    with st.expander(
                        "View Grad-CAM response"
                    ):

                        st.json(
                            normalize_result_for_json(
                                st.session_state[
                                    "gradcam_result"
                                ]
                            )
                        )

            # -----------------------------------------------
            # WHY GRAD-CAM
            # -----------------------------------------------

            st.divider()

            st.subheader(
                "Why Grad-CAM?"
            )

            st.write(
                """
                Prediction probability indicates how strongly
                the model favors a class, but it does not show
                where the model focused.

                Grad-CAM adds an interpretability layer by
                highlighting spatial regions of the X-ray that
                contributed to the prediction.

                This makes the research prototype easier to
                inspect and evaluate rather than treating the
                prediction as an unexplained classification score.
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
        The project is implemented as a deployable machine
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

    d1, d2 = st.columns(2)

    with d1:

        st.write(
            "**Model Repository**"
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
            "**Experiment**"
        )

        st.code(
            MODEL_EXPERIMENT
        )

    with d2:

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

        <strong>
            AI Knee Osteoarthritis Detection
        </strong>

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
