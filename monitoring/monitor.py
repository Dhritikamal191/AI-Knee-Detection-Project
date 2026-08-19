from pathlib import Path
import pandas as pd


LOG_FILE = Path(
    "artifacts/logs/prediction_logs.csv"
)


def load_prediction_logs():

    if not LOG_FILE.exists():

        return pd.DataFrame(
            columns=[
                "timestamp",
                "filename",
                "prediction",
                "confidence",
                "device"
            ]
        )

    return pd.read_csv(
        LOG_FILE
    )


def get_monitoring_summary():

    df = load_prediction_logs()

    if df.empty:

        return {
            "total_predictions": 0,
            "average_confidence": None,
            "grade_distribution": {},
        }

    return {
        "total_predictions": len(df),

        "average_confidence": round(
            df["confidence"].mean(),
            4
        ),

        "grade_distribution": (
            df["prediction"]
            .value_counts()
            .to_dict()
        )
    }


if __name__ == "__main__":

    summary = get_monitoring_summary()

    print("\n" + "=" * 60)
    print("MODEL MONITORING SUMMARY")
    print("=" * 60)

    print(
        f"Total predictions: "
        f"{summary['total_predictions']}"
    )

    print(
        f"Average confidence: "
        f"{summary['average_confidence']}"
    )

    print("\nGrade distribution:")

    for grade, count in (
        summary["grade_distribution"].items()
    ):

        print(
            f"Grade {grade}: {count}"
        )