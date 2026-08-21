from pydantic import BaseModel, Field
from typing import Dict


class HealthResponse(BaseModel):
    status: str
    model: str


class PredictionResponse(BaseModel):
    prediction: str = Field(
        ...,
        description="Predicted osteoarthritis grade"
    )

    prediction_index: int = Field(
        ...,
        ge=0,
        le=4,
        description="Predicted grade index from 0 to 4"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence probability"
    )

    confidence_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Confidence percentage"
    )

    threshold_probabilities: Dict[str, float]

    grade_probabilities: Dict[str, Dict[str, float]]