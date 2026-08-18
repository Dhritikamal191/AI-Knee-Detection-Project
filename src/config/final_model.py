from pathlib import Path


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

FINAL_MODEL_NAME = "Experiment 5"

FINAL_CHECKPOINT = (
    Path("artifacts/checkpoints")
    / "best_model_experiment5.pt"
)

FINAL_METRICS = (
    Path("artifacts/metrics")
    / "experiment5_metrics.json"
)

FINAL_EXPLAINABILITY = (
    Path("artifacts/evaluation")
    / "explainability"
)

FINAL_ERROR_ANALYSIS = (
    Path("artifacts/evaluation")
    / "error_analysis"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_ARCHITECTURE = "ResNet50"

MODEL_TYPE = "Ordinal ResNet50"

NUM_CLASSES = 5

CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4"
]

ORDINAL_THRESHOLDS = [
    ">=1",
    ">=2",
    ">=3",
    ">=4"
]


# ============================================================
# VALIDATION
# ============================================================

def validate_final_model():

    if not FINAL_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Final model checkpoint not found:\n"
            f"{FINAL_CHECKPOINT}"
        )

    if not FINAL_METRICS.exists():

        raise FileNotFoundError(
            f"Final model metrics not found:\n"
            f"{FINAL_METRICS}"
        )

    return True


if __name__ == "__main__":

    validate_final_model()

    print("=" * 70)
    print("FINAL MODEL CONFIGURATION")
    print("=" * 70)

    print(
        f"Model:      {FINAL_MODEL_NAME}"
    )

    print(
        f"Architecture: {MODEL_ARCHITECTURE}"
    )

    print(
        f"Checkpoint: {FINAL_CHECKPOINT}"
    )

    print(
        f"Metrics:    {FINAL_METRICS}"
    )

    print(
        f"Classes:    {CLASSES}"
    )

    print(
        f"Thresholds: {ORDINAL_THRESHOLDS}"
    )

    print()
    print("Final model configuration validated.")