import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

METRICS_DIR = Path("artifacts/metrics")

EXP5_FILE = METRICS_DIR / "experiment5_metrics.json"
EXP6_FILE = METRICS_DIR / "experiment6_metrics.json"

OUTPUT_FILE = (
    METRICS_DIR /
    "experiment_comparison.json"
)


# ============================================================
# LOAD METRICS
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    with open(
        path,
        "r"
    ) as f:

        return json.load(f)


# ============================================================
# GET METRIC
# ============================================================

def get_metric(
    data,
    *keys
):

    for key in keys:

        if key in data:

            return data[key]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "EXPERIMENT 5 vs EXPERIMENT 6"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    exp5 = load_json(
        EXP5_FILE
    )

    exp6 = load_json(
        EXP6_FILE
    )


    # ========================================================
    # METRICS
    # ========================================================

    metric_names = [

        (
            "Accuracy",
            ["test_accuracy", "accuracy"]
        ),

        (
            "Macro F1",
            ["test_macro_f1", "macro_f1"]
        ),

        (
            "Balanced Accuracy",
            [
                "test_balanced_accuracy",
                "balanced_accuracy"
            ]
        ),

        (
            "Quadratic Weighted Kappa",
            [
                "test_qwk",
                "quadratic_weighted_kappa"
            ]
        ),

        (
            "Mean Absolute Error",
            [
                "test_mae",
                "mean_absolute_error"
            ]
        )
    ]


    comparison = {}


    # ========================================================
    # PRINT COMPARISON
    # ========================================================

    print()

    print(
        f"{'Metric':<28}"
        f"{'Experiment 5':>18}"
        f"{'Experiment 6':>18}"
        f"{'Better':>12}"
    )

    print("-" * 76)


    for name, keys in metric_names:

        value5 = get_metric(
            exp5,
            *keys
        )

        value6 = get_metric(
            exp6,
            *keys
        )


        if value5 is None:
            value5 = 0.0

        if value6 is None:
            value6 = 0.0


        # MAE is lower-is-better
        if name == "Mean Absolute Error":

            if value5 < value6:

                better = "Exp 5"

            elif value6 < value5:

                better = "Exp 6"

            else:

                better = "Tie"

        else:

            if value5 > value6:

                better = "Exp 5"

            elif value6 > value5:

                better = "Exp 6"

            else:

                better = "Tie"


        comparison[name] = {

            "experiment5": value5,

            "experiment6": value6,

            "difference_exp6_minus_exp5":
                value6 - value5,

            "better": better
        }


        print(

            f"{name:<28}"
            f"{value5:>18.4f}"
            f"{value6:>18.4f}"
            f"{better:>12}"
        )


    # ========================================================
    # OVERALL DECISION
    # ========================================================

    exp5_wins = 0
    exp6_wins = 0


    for metric in comparison.values():

        if metric["better"] == "Exp 5":

            exp5_wins += 1

        elif metric["better"] == "Exp 6":

            exp6_wins += 1


    if exp5_wins > exp6_wins:

        final_model = "Experiment 5"

    elif exp6_wins > exp5_wins:

        final_model = "Experiment 6"

    else:

        # QWK is particularly important
        # for ordinal grading.

        qwk = comparison[
            "Quadratic Weighted Kappa"
        ]

        if (
            qwk["experiment5"]
            >
            qwk["experiment6"]
        ):

            final_model = "Experiment 5"

        else:

            final_model = "Experiment 6"


    # ========================================================
    # FINAL DECISION
    # ========================================================

    if final_model == "Experiment 5":

        checkpoint = (
            "artifacts/checkpoints/"
            "best_model_experiment5.pt"
        )

    else:

        checkpoint = (
            "artifacts/checkpoints/"
            "best_model_experiment6.pt"
        )


    print()

    print("=" * 70)

    print(
        f"Experiment 5 wins: {exp5_wins}"
    )

    print(
        f"Experiment 6 wins: {exp6_wins}"
    )

    print()

    print(
        f"FINAL MODEL: {final_model}"
    )

    print(
        f"Checkpoint: {checkpoint}"
    )

    print("=" * 70)


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = {

        "experiment5": {
            "metrics_file":
                str(EXP5_FILE)
        },

        "experiment6": {
            "metrics_file":
                str(EXP6_FILE)
        },

        "comparison":
            comparison,

        "experiment5_wins":
            exp5_wins,

        "experiment6_wins":
            exp6_wins,

        "final_model":
            final_model,

        "final_checkpoint":
            checkpoint
    }


    with open(
        OUTPUT_FILE,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )


    print()

    print(
        "Comparison saved:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()