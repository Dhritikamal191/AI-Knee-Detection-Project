import io
import json
import tempfile
from pathlib import Path
from textwrap import dedent
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.inference.predict import predict
from src.explainability.gradcam import generate_gradcam


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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 8% 0%,
                rgba(99, 102, 241, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 92% 8%,
                rgba(59, 130, 246, 0.09),
                transparent 25%
            ),
            linear-gradient(
                180deg,
                #08090d 0%,
                #090a0f 55%,
                #08090d 100%
            );

        color: #f4f4f5;
    }


    .block-container {
        max-width: 1420px;

        padding-top: 2rem;
        padding-bottom: 5rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* Remove unnecessary Streamlit spacing */

    div[data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1,
    h2,
    h3,
    h4 {
        color: #f4f4f5 !important;

        letter-spacing: -0.035em;
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


    p {
        color: #a1a5b0;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0b0d13 0%,
                #090a0f 50%,
                #08090d 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.07);
    }


    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }


    section[data-testid="stSidebar"] * {
        color: #d7d9e0;
    }


    /* Sidebar radio */

    section[data-testid="stSidebar"]
    div[role="radiogroup"] {

        gap: 4px;
    }


    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {

        border-radius: 10px;

        padding: 7px 10px;

        transition:
            background 0.2s ease,
            transform 0.2s ease;
    }


    section[data-testid="stSidebar"]
    div[role="radiogroup"] label:hover {

        background:
            rgba(255,255,255,0.045);
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {

        padding:
            25px
            0
            35px
            0;
    }


    .hero-badge {

        display: inline-flex;

        align-items: center;

        padding:
            7px
            13px;

        border-radius: 999px;

        background:
            rgba(99,102,241,0.10);

        border:
            1px solid
            rgba(129,140,248,0.22);

        color:
            #a5b4fc;

        font-size:
            11px;

        font-weight:
            750;

        letter-spacing:
            0.10em;

        text-transform:
            uppercase;

        box-shadow:
            0 0 25px
            rgba(99,102,241,0.06);
    }


    .hero-title {

        margin-top:
            18px;

        max-width:
            1000px;

        font-size:
            clamp(42px, 6vw, 76px);

        line-height:
            0.98;

        font-weight:
            900;

        letter-spacing:
            -0.06em;

        background:
            linear-gradient(
                100deg,
                #ffffff 8%,
                #c7d2fe 46%,
                #93c5fd 85%
            );

        -webkit-background-clip:
            text;

        background-clip:
            text;

        -webkit-text-fill-color:
            transparent;
    }


    .hero-subtitle {

        max-width:
            850px;

        margin-top:
            22px;

        color:
            #9297a5;

        font-size:
            16px;

        line-height:
            1.85;
    }


    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-title {

        margin-top:
            42px;

        margin-bottom:
            12px;

        font-size:
            28px;

        line-height:
            1.2;

        font-weight:
            800;

        letter-spacing:
            -0.04em;

        color:
            #f4f4f5;
    }


    .section-description {

        max-width:
            850px;

        color:
            #858a97;

        line-height:
            1.8;

        margin-bottom:
            24px;
    }


    /* ========================================================
       GENERAL GLASS CARDS
       ======================================================== */

    .glass-card {

        padding:
            24px;

        height:
            100%;

        box-sizing:
            border-box;

        border-radius:
            18px;

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.050),
                rgba(255,255,255,0.014)
            );

        border:
            1px solid
            rgba(255,255,255,0.075);

        box-shadow:
            inset 0 1px 0
            rgba(255,255,255,0.025),
            0 10px 35px
            rgba(0,0,0,0.10);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            background 0.2s ease;
    }


    .glass-card:hover {

        transform:
            translateY(-2px);

        border-color:
            rgba(129,140,248,0.20);

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.060),
                rgba(255,255,255,0.020)
            );
    }


    .card-label {

        color:
            #737987;

        font-size:
            10px;

        font-weight:
            750;

        letter-spacing:
            0.13em;

        text-transform:
            uppercase;

        margin-bottom:
            8px;
    }


    .card-value {

        color:
            #f1f2f5;

        font-size:
            28px;

        line-height:
            1.15;

        font-weight:
            800;

        letter-spacing:
            -0.035em;
    }


    .card-description {

        color:
            #777c89;

        font-size:
            12px;

        line-height:
            1.55;

        margin-top:
            7px;
    }


    /* ========================================================
       KPI / METRIC CARDS
       ======================================================== */

    .metric-card {

        padding:
            22px;

        min-height:
            120px;

        height:
            100%;

        box-sizing:
            border-box;

        display:
            flex;

        flex-direction:
            column;

        justify-content:
            center;

        border-radius:
            18px;

        background:
            linear-gradient(
                145deg,
                rgba(99,102,241,0.095),
                rgba(255,255,255,0.018)
            );

        border:
            1px solid
            rgba(255,255,255,0.08);

        box-shadow:
            inset 0 1px 0
            rgba(255,255,255,0.025),
            0 12px 30px
            rgba(0,0,0,0.10);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }


    .metric-card:hover {

        transform:
            translateY(-3px);

        border-color:
            rgba(129,140,248,0.25);
    }


    .metric-number {

        font-size:
            32px;

        line-height:
            1.1;

        font-weight:
            850;

        letter-spacing:
            -0.045em;

        color:
            #f4f4f5;

        white-space:
            nowrap;
    }


    .metric-label {

        margin-top:
            8px;

        color:
            #858a98;

        font-size:
            10px;

        line-height:
            1.5;

        font-weight:
            700;

        text-transform:
            uppercase;

        letter-spacing:
            0.09em;
    }


    /* ========================================================
       PREDICTION CARD
       ======================================================== */

    .prediction-card {

        padding:
            30px;

        min-height:
            230px;

        border-radius:
            22px;

        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(99,102,241,0.13),
                transparent 40%
            ),
            linear-gradient(
                145deg,
                rgba(99,102,241,0.11),
                rgba(255,255,255,0.025)
            );

        border:
            1px solid
            rgba(129,140,248,0.22);

        box-shadow:
            inset 0 1px 0
            rgba(255,255,255,0.035),
            0 20px 50px
            rgba(0,0,0,0.15);
    }


    .prediction-grade {

        margin:
            8px
            0
            10px
            0;

        font-size:
            clamp(48px, 6vw, 68px);

        line-height:
            1;

        font-weight:
            900;

        letter-spacing:
            -0.065em;

        background:
            linear-gradient(
                100deg,
                #ffffff,
                #a5b4fc,
                #93c5fd
            );

        -webkit-background-clip:
            text;

        background-clip:
            text;

        -webkit-text-fill-color:
            transparent;
    }


    .prediction-confidence {

        color:
            #a1a5b0;

        font-size:
            15px;

        font-weight:
            600;
    }


    /* ========================================================
       PROBABILITY DISTRIBUTION
       ======================================================== */

    .probability-row {

        margin-bottom:
            17px;
    }


    .probability-header {

        display:
            flex;

        align-items:
            center;

        justify-content:
            space-between;

        margin-bottom:
            7px;

        font-size:
            13px;
    }


    .probability-label {

        color:
            #a4a8b2;

        font-weight:
            550;
    }


    .probability-value {

        color:
            #f1f1f3;

        font-weight:
            700;
    }


    .probability-track {

        width:
            100%;

        height:
            8px;

        overflow:
            hidden;

        border-radius:
            999px;

        background:
            rgba(255,255,255,0.065);

        box-shadow:
            inset 0 1px 2px
            rgba(0,0,0,0.20);
    }


    .probability-fill {

        height:
            100%;

        border-radius:
            inherit;

        background:
            linear-gradient(
                90deg,
                #6366f1,
                #818cf8,
                #60a5fa
            );

        box-shadow:
            0 0 12px
            rgba(99,102,241,0.18);

        transition:
            width 0.5s ease;
    }


    /* ========================================================
       UPLOADED IMAGE
       ======================================================== */

    .xray-card {

        padding:
            14px;

        border-radius:
            20px;

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            rgba(255,255,255,0.08);

        box-shadow:
            0 15px 40px
            rgba(0,0,0,0.15);
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {

        padding:
            8px;

        border-radius:
            18px;

        background:
            rgba(255,255,255,0.018);

        border:
            1px dashed
            rgba(255,255,255,0.12);

        transition:
            border-color 0.2s ease,
            background 0.2s ease;
    }


    [data-testid="stFileUploader"]:hover {

        border-color:
            rgba(129,140,248,0.35);

        background:
            rgba(99,102,241,0.025);
    }


    [data-testid="stFileUploaderDropzone"] {

        background:
            transparent;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        min-height:
            44px;

        padding:
            0
            18px;

        border-radius:
            11px;

        border:
            1px solid
            rgba(129,140,248,0.22);

        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,0.18),
                rgba(59,130,246,0.10)
            );

        color:
            #eef0ff;

        font-size:
            14px;

        font-weight:
            700;

        box-shadow:
            0 8px 25px
            rgba(0,0,0,0.12);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            background 0.2s ease;
    }


    .stButton > button:hover {

        transform:
            translateY(-1px);

        border-color:
            rgba(129,140,248,0.50);

        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,0.28),
                rgba(59,130,246,0.16)
            );

        color:
            #ffffff;
    }


    .stButton > button:active {

        transform:
            translateY(0);
    }


    /* ========================================================
       SELECT / RADIO / INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div {

        background:
            rgba(255,255,255,0.025);

        border-color:
            rgba(255,255,255,0.10);

        border-radius:
            10px;
    }


    div[data-baseweb="select"] > div:hover {

        border-color:
            rgba(129,140,248,0.35);
    }


    /* ========================================================
       DATAFRAME / TABLE
       ======================================================== */

    [data-testid="stDataFrame"] {

        border:
            1px solid
            rgba(255,255,255,0.08);

        border-radius:
            14px;

        overflow:
            hidden;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {

        border-radius:
            12px;

        border:
            1px solid
            rgba(255,255,255,0.08);

        background:
            rgba(255,255,255,0.025);
    }


    /* ========================================================
       METRIC / THRESHOLD GRID
       ======================================================== */

    .threshold-card {

        padding:
            20px;

        min-height:
            105px;

        border-radius:
            16px;

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            rgba(255,255,255,0.075);
    }


    /* ========================================================
       CODE BLOCKS
       ======================================================== */

    pre {

        border-radius:
            12px !important;

        border:
            1px solid
            rgba(255,255,255,0.08) !important;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    details {

        border-radius:
            12px !important;

        border-color:
            rgba(255,255,255,0.08) !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {

        border-color:
            rgba(255,255,255,0.07) !important;

        margin:
            28px
            0;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {

        margin-top:
            75px;

        padding-top:
            28px;

        padding-bottom:
            10px;

        border-top:
            1px solid
            rgba(255,255,255,0.07);

        text-align:
            center;

        color:
            #626775;

        font-size:
            11px;

        line-height:
            1.7;
    }


    /* ========================================================
       GRAD-CAM VISUALIZATION
       ======================================================== */

    .explainability-card {

        padding:
            16px;

        border-radius:
            18px;

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            rgba(255,255,255,0.08);

        box-shadow:
            0 12px 35px
            rgba(0,0,0,0.12);
    }


    .explainability-title {

        margin-bottom:
            10px;

        color:
            #d7d9e0;

        font-size:
            13px;

        font-weight:
            700;

        text-align:
            center;
    }


    /* ========================================================
       STATUS INDICATORS
       ======================================================== */

    .status-online {

        display:
            inline-flex;

        align-items:
            center;

        gap:
            7px;

        color:
            #a7f3d0;

        font-size:
            12px;

        font-weight:
            650;
    }


    .status-dot {

        width:
            7px;

        height:
            7px;

        border-radius:
            50%;

        background:
            #34d399;

        box-shadow:
            0 0 10px
            rgba(52,211,153,0.55);
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 1100px) {

        .block-container {

            padding-left:
                2rem;

            padding-right:
                2rem;
        }

        .metric-number {

            font-size:
                28px;
        }

        .metric-label {

            font-size:
                9px;
        }
    }


    @media (max-width: 768px) {

        .block-container {

            padding-left:
                1rem;

            padding-right:
                1rem;

            padding-top:
                1rem;
        }


        .hero {

            padding-top:
                15px;

            padding-bottom:
                25px;
        }


        .hero-title {

            font-size:
                46px;

            line-height:
                1.0;
        }


        .hero-subtitle {

            font-size:
                14px;

            line-height:
                1.7;
        }


        .metric-card {

            min-height:
                105px;

            padding:
                17px;
        }


        .metric-number {

            font-size:
                25px;
        }


        .prediction-card {

            padding:
                24px;
        }


        .prediction-grade {

            font-size:
                50px;
        }


        .section-title {

            font-size:
                24px;
        }
    }


    @media (max-width: 500px) {

        .hero-title {

            font-size:
                38px;
        }


        .hero-badge {

            font-size:
                9px;

            letter-spacing:
                0.07em;
        }


        .metric-number {

            font-size:
                22px;
        }


        .metric-label {

            font-size:
                8px;
        }


        .card-value {

            font-size:
                24px;
        }
    }


    /* ========================================================
       STREAMLIT UI CLEANUP
       ======================================================== */

    #MainMenu {
        visibility:
            hidden;
    }


    footer {
        visibility:
            hidden;
    }


    header[data-testid="stHeader"] {

        background:
            transparent;
    }


    /* Prevent accidental horizontal overflow */

    .main,
    .block-container {

        overflow-x:
            hidden;
    }


    </style>
    """,
    unsafe_allow_html=True,
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
    2: "Mild to moderate osteoarthritis",
    3: "Moderate to severe osteoarthritis",
    4: "Severe osteoarthritis",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_result_value(result, key, default=None):
    """Safely retrieve values from prediction result."""

    if isinstance(result, dict):
        return result.get(key, default)

    return getattr(result, key, default)


def normalize_probabilities(result):
    """
    Extract class probabilities from the prediction result.

    Supports either:
        class_probabilities
        grade_probabilities
        probabilities
    """

    probabilities = (
        get_result_value(result, "class_probabilities")
        or get_result_value(result, "grade_probabilities")
        or get_result_value(result, "probabilities")
    )

    if probabilities is None:
        return None

    if isinstance(probabilities, dict):

        values = []

        for grade in GRADE_NAMES:
            value = probabilities.get(grade, 0)

            values.append(float(value))

        return values

    return [float(x) for x in probabilities]


def get_threshold_probabilities(result):
    """Extract ordinal threshold probabilities."""

    values = (
        get_result_value(
            result,
            "ordinal_threshold_probabilities"
        )
        or get_result_value(
            result,
            "threshold_probabilities"
        )
    )

    if values is None:
        return None

    if isinstance(values, dict):
        return values

    return {
        f"Grade >= {i + 1}": float(value)
        for i, value in enumerate(values)
    }


def probability_percent(value):
    """Convert probability to percentage."""

    value = float(value)

    if value <= 1:
        return value * 100

    return value


def render_probability_bars(probabilities):

    for grade, probability in zip(
        GRADE_NAMES,
        probabilities
    ):

        percent = probability_percent(probability)

        percent = max(0, min(100, percent))

        st.markdown(
            f"""
            <div class="probability-row">

                <div class="probability-header">

                    <span class="probability-label">
                        {grade}
                    </span>

                    <span class="probability-value">
                        {percent:.2f}%
                    </span>

                </div>

                <div class="probability-track">

                    <div
                        class="probability-fill"
                        style="width:{percent}%;">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


def save_uploaded_image(uploaded_file):

    suffix = Path(
        uploaded_file.name
    ).suffix or ".png"

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(
        uploaded_file.getbuffer()
    )

    temp_file.close()

    return Path(temp_file.name)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:850;
            letter-spacing:-0.04em;
            margin-bottom:4px;
        ">
            🦴 Knee AI
        </div>

        <div style="
            color:#777c89;
            font-size:12px;
            margin-bottom:25px;
        ">
            Osteoarthritis Detection
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown("### Navigation")

    page = st.radio(
        "Navigate",
        [
            "Prediction",
            "Model",
            "Explainability",
            "System",
        ],
        label_visibility="collapsed",
    )


    st.markdown("---")


    st.markdown("### Model")

    st.caption(
        "Experiment 5 · Ordinal ResNet50"
    )

    st.caption(
        "5 severity grades"
    )

    st.caption(
        "224 × 224 RGB input"
    )


    st.markdown("---")


    st.markdown(
        """
        <div style="
            color:#626775;
            font-size:11px;
            line-height:1.6;
        ">
            Research / educational prototype.
            Not intended for independent medical
            diagnosis or treatment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            Computer Vision · Deep Learning · MLOps
        </div>

        <div class="hero-title">
            AI Knee Osteoarthritis Detection
        </div>

        <div class="hero-subtitle">

            Upload a knee X-ray and obtain an automated
            osteoarthritis severity prediction using an
            Ordinal ResNet50 model, with confidence scores,
            probability distributions and explainable AI.

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

metric_cols = st.columns(4)

metrics = [
    ("60.93%", "Accuracy"),
    ("59.89%", "Macro F1"),
    ("77.07%", "Quadratic Weighted Kappa"),
    ("5", "Severity Grades"),
]

for col, (value, label) in zip(
    metric_cols,
    metrics
):

    with col:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {value}
                </div>

                <div class="metric-label">
                    {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "Prediction":

    st.markdown(
        '<div class="section-title">Knee X-ray Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">

            Upload a knee X-ray image. The model will preprocess
            the image and return the predicted osteoarthritis
            severity together with probability estimates.

        </div>
        """,
        unsafe_allow_html=True,
    )


    uploaded_file = st.file_uploader(
        "Upload knee X-ray",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        help="Upload a knee X-ray image for analysis.",
    )


    if uploaded_file is None:

        st.info(
            "Upload a knee X-ray image to begin."
        )

        st.markdown(
            """
            <div class="glass-card">

                <div class="card-label">
                    Supported workflow
                </div>

                <div style="
                    font-size:18px;
                    font-weight:700;
                    margin-bottom:8px;
                ">
                    Image → Model → Severity → Explanation
                </div>

                <div style="
                    color:#777c89;
                    font-size:13px;
                ">
                    The application combines ordinal deep learning,
                    probability estimation and Grad-CAM explainability.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    else:

        image = Image.open(uploaded_file).convert("RGB")

        image_col, info_col = st.columns(
            [1.05, 0.95],
            gap="large",
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with image_col:

            st.markdown(
                "### Uploaded X-ray"
            )

            st.image(
                image,
                width="stretch",
            )

            st.caption(
                f"{uploaded_file.name} · "
                f"{image.width} × {image.height}px"
            )


        # ----------------------------------------------------
        # RUN INFERENCE
        # ----------------------------------------------------

        with info_col:

            st.markdown(
                "### Analysis"
            )

            analyze = st.button(
                "Run AI Analysis",
                width="stretch",
            )


            if analyze:

                with st.spinner(
                    "Running model inference..."
                ):

                    image_path = save_uploaded_image(
                        uploaded_file
                    )

                    try:

                        result = predict(
                            image_path
                        )

                    except Exception as exc:

                        st.error(
                            f"Inference failed: {exc}"
                        )

                        st.stop()


                st.session_state[
                    "prediction_result"
                ] = result

                st.session_state[
                    "prediction_image"
                ] = str(image_path)


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        result = st.session_state.get(
            "prediction_result"
        )


        if result is not None:

            predicted_grade = get_result_value(
                result,
                "predicted_grade",
                get_result_value(
                    result,
                    "grade",
                    0
                ),
            )

            confidence = get_result_value(
                result,
                "confidence",
                0
            )


            try:
                predicted_grade = int(
                    predicted_grade
                )
            except Exception:
                predicted_grade = 0


            confidence = probability_percent(
                confidence
            )


            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )


            st.markdown(
                f"""
                <div class="prediction-card">

                    <div class="card-label">
                        Model Prediction
                    </div>

                    <div class="prediction-grade">
                        Grade {predicted_grade}
                    </div>

                    <div class="prediction-confidence">
                        Confidence · {confidence:.2f}%
                    </div>

                    <div style="
                        margin-top:14px;
                        color:#777c89;
                        font-size:13px;
                    ">
                        {GRADE_DESCRIPTIONS.get(
                            predicted_grade,
                            "Severity classification"
                        )}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            probabilities = normalize_probabilities(
                result
            )


            if probabilities:

                st.markdown(
                    "### Class Probability Distribution"
                )

                render_probability_bars(
                    probabilities
                )


            # ------------------------------------------------
            # THRESHOLDS
            # ------------------------------------------------

            threshold_probs = (
                get_threshold_probabilities(
                    result
                )
            )


            if threshold_probs:

                st.markdown(
                    "### Ordinal Threshold Probabilities"
                )

                threshold_cols = st.columns(
                    len(threshold_probs)
                )


                for col, (
                    name,
                    value
                ) in zip(
                    threshold_cols,
                    threshold_probs.items()
                ):

                    with col:

                        percent = probability_percent(
                            value
                        )

                        st.markdown(
                            f"""
                            <div class="glass-card">

                                <div class="card-label">
                                    {name}
                                </div>

                                <div class="card-value">
                                    {percent:.2f}%
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# ============================================================
# MODEL PAGE
# ============================================================

elif page == "Model":

    st.markdown(
        '<div class="section-title">Model Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">

            Experiment 5 uses an Ordinal ResNet50 architecture.
            The model predicts five ordered osteoarthritis severity
            levels through four ordinal thresholds.

        </div>
        """,
        unsafe_allow_html=True,
    )


    cols = st.columns(2)


    with cols[0]:

        st.markdown(
            """
            <div class="glass-card">

                <div class="card-label">
                    Architecture
                </div>

                <div class="card-value">
                    Ordinal ResNet50
                </div>

                <div class="card-description">
                    Deep residual network with ordinal output formulation.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with cols[1]:

        st.markdown(
            """
            <div class="glass-card">

                <div class="card-label">
                    Input
                </div>

                <div class="card-value">
                    224 × 224 RGB
                </div>

                <div class="card-description">
                    Image preprocessing with ImageNet normalization.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        "### Severity Classes"
    )


    class_cols = st.columns(5)

    for i, col in enumerate(class_cols):

        with col:

            st.markdown(
                f"""
                <div class="glass-card">

                    <div class="card-label">
                        Class
                    </div>

                    <div class="card-value">
                        {i}
                    </div>

                    <div class="card-description">
                        {GRADE_NAMES[i]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    st.markdown(
        "### Model Performance"
    )


    performance_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro F1",
                "Quadratic Weighted Kappa",
            ],
            "Score": [
                0.6093,
                0.5989,
                0.7707,
            ],
        }
    )


    performance_df["Score"] = (
        performance_df["Score"] * 100
    ).round(2)


    st.dataframe(
        performance_df,
        hide_index=True,
        width="stretch",
    )


# ============================================================
# EXPLAINABILITY PAGE
# ============================================================

elif page == "Explainability":

    st.markdown(
        '<div class="section-title">Explainable AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">

            Grad-CAM is used to visualize image regions that
            contribute to the model's prediction. This provides
            an additional inspection layer beyond the numerical
            prediction.

        </div>
        """,
        unsafe_allow_html=True,
    )


    result = st.session_state.get(
        "prediction_result"
    )

    image_path = st.session_state.get(
        "prediction_image"
    )


    if result is None or image_path is None:

        st.info(
            "Run a prediction first to generate a Grad-CAM explanation."
        )

    else:

        generate = st.button(
            "Generate Grad-CAM Explanation",
            width="stretch",
        )


        if generate:

            with st.spinner(
                "Generating Grad-CAM visualization..."
            ):

                try:

                    gradcam_result = generate_gradcam(
                        image_path
                    )

                    st.session_state[
                        "gradcam_result"
                    ] = gradcam_result

                except Exception as exc:

                    st.error(
                        f"Grad-CAM generation failed: {exc}"
                    )


        gradcam_result = st.session_state.get(
            "gradcam_result"
        )


        if gradcam_result:

            st.markdown(
                "### Model Attention"
            )


            # Support dictionary-style Grad-CAM results

            if isinstance(
                gradcam_result,
                dict
            ):

                original = (
                    gradcam_result.get(
                        "original"
                    )
                )

                overlay = (
                    gradcam_result.get(
                        "overlay"
                    )
                )

                heatmap = (
                    gradcam_result.get(
                        "heatmap"
                    )
                )


                visual_cols = st.columns(
                    3
                )


                for col, (
                    title,
                    path
                ) in zip(
                    visual_cols,
                    [
                        ("Original", original),
                        ("Heatmap", heatmap),
                        ("Overlay", overlay),
                    ],
                ):

                    with col:

                        st.markdown(
                            f"**{title}**"
                        )

                        if path and Path(
                            path
                        ).exists():

                            st.image(
                                str(path),
                                width="stretch",
                            )

                        else:

                            st.info(
                                "Visualization unavailable."
                            )

            else:

                st.info(
                    "Grad-CAM generated successfully."
                )


# ============================================================
# SYSTEM PAGE
# ============================================================

elif page == "System":

    st.markdown(
        '<div class="section-title">System Architecture</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="section-description">

            The application separates the presentation layer,
            inference service and model artifact, allowing the
            same model to be consumed locally, through the API
            or through the Streamlit interface.

        </div>
        """,
        unsafe_allow_html=True,
    )


    architecture_cols = st.columns(4)


    architecture = [
        (
            "01",
            "Streamlit",
            "Interactive user interface",
        ),
        (
            "02",
            "FastAPI",
            "REST inference service",
        ),
        (
            "03",
            "Ordinal ResNet50",
            "Five-grade prediction model",
        ),
        (
            "04",
            "Hugging Face",
            "Versioned model artifact",
        ),
    ]


    for col, (
        number,
        title,
        description,
    ) in zip(
        architecture_cols,
        architecture
    ):

        with col:

            st.markdown(
                f"""
                <div class="glass-card">

                    <div class="card-label">
                        {number}
                    </div>

                    <div style="
                        font-size:20px;
                        font-weight:800;
                        margin-bottom:7px;
                    ">
                        {title}
                    </div>

                    <div style="
                        color:#777c89;
                        font-size:12px;
                        line-height:1.6;
                    ">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    st.markdown(
        "### Engineering Stack"
    )


    technologies = [
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
        "Hugging Face Hub",
        "MLflow",
    ]


    st.write(
        " · ".join(technologies)
    )


    st.markdown(
        "### Model Artifact"
    )


    st.code(
        "SHA256: "
        "f2200b43966dce1498e47ad6ee45cb35e5cec831246b7667323e16fd9d7e1667",
        language="text",
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>Research / Educational Prototype</strong>
        <br><br>

        This application is not a certified medical device and
        should not be used independently for diagnosis or treatment
        decisions. Model predictions and Grad-CAM visualizations
        are intended for research and demonstration purposes.

        <br><br>

        AI Knee Osteoarthritis Detection · Experiment 5 ·
        Ordinal ResNet50

    </div>
    """,
    unsafe_allow_html=True,
)
