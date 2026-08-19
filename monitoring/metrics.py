from pathlib import Path
from datetime import datetime
import csv


LOG_DIR = Path("artifacts/logs")
LOG_FILE = LOG_DIR / "prediction_logs.csv"


def log_prediction(
    filename,
    prediction,
    confidence,
    device="unknown"
):
    """
    Log a model prediction for monitoring.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = LOG_FILE.exists()

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "filename",
                "prediction",
                "confidence",
                "device"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            filename,
            prediction,
            confidence,
            device
        ])


def get_log_path():
    return LOG_FILE