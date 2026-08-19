from pathlib import Path
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from src.inference.predict import predict


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Knee Osteoarthritis Grading API",
    description=(
        "API for AI-assisted knee osteoarthritis grading "
        "using an Ordinal ResNet50 model."
    ),
    version="1.0.0",
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Knee Osteoarthritis Grading API",
        "model": "Experiment 5 - Ordinal ResNet50",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, or PNG."
            )
        )

    try:

        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as tmp:

            tmp.write(contents)
            temp_path = Path(tmp.name)

        # Validate image
        Image.open(
            temp_path
        ).verify()

        # Run model inference
        result = predict(
            temp_path
        )

        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "prediction_index": result["prediction_index"],
            "confidence": result["confidence"],
            "confidence_percent": result["confidence_percent"],
            "threshold_probabilities": (
                result["threshold_probabilities"]
            ),
            "grade_probabilities": (
                result["grade_probabilities"]
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:

        if "temp_path" in locals():
            temp_path.unlink(
                missing_ok=True
            )
