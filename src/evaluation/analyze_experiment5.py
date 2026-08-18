import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
OUTPUT_DIR = BASE_DIR / "artifacts" / "evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_FILE = (
    METRICS_DIR /
    "experiment5_metrics.json"
)


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics():

    if not METRICS_FILE.exists():

        raise FileNotFoundError(
            f"Metrics file not found:\n{METRICS_FILE}"
        )

    with open(
        METRICS_FILE,
        "r"
    ) as f:

        return json.load(f)


# ============================================================
# FIND CONFUSION MATRIX
# ============================================================

def find_confusion_matrix(data):

    """
    Recursively searches the metrics JSON
    for a 5 x 5 confusion matrix.
    """

    def search(obj):

        if isinstance(obj, dict):

            for key, value in obj.items():

                key_lower = str(key).lower()

                if (
                    "confusion" in key_lower
                    and isinstance(value, list)
                ):

                    return value

                result = search(value)

                if result is not None:

                    return result

        elif isinstance(obj, list):

            for item in obj:

                result = search(item)

                if result is not None:

                    return result

        return None

    return search(data)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(cm):

    cm = np.array(
        cm,
        dtype=int
    )

    classes = [
        "Grade 0",
        "Grade 1",
        "Grade 2",
        "Grade 3",
        "Grade 4"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        cm
    )

    ax.set_title(
        "Experiment 5 - Confusion Matrix",
        fontsize=15
    )

    ax.set_xlabel(
        "Predicted Grade"
    )

    ax.set_ylabel(
        "Actual Grade"
    )

    ax.set_xticks(
        np.arange(5)
    )

    ax.set_yticks(
        np.arange(5)
    )

    ax.set_xticklabels(
        classes
    )

    ax.set_yticklabels(
        classes
    )

    # Write values inside cells

    for i in range(5):

        for j in range(5):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12
            )

    plt.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "confusion_matrix_experiment5.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved:\n"
        f"{output_path}"
    )


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

def create_normalized_confusion_matrix(cm):

    cm = np.array(
        cm,
        dtype=float
    )

    row_totals = cm.sum(
        axis=1,
        keepdims=True
    )

    normalized = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm),
        where=row_totals != 0
    )

    classes = [
        "Grade 0",
        "Grade 1",
        "Grade 2",
        "Grade 3",
        "Grade 4"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        normalized
    )

    ax.set_title(
        "Experiment 5 - Normalized Confusion Matrix",
        fontsize=15
    )

    ax.set_xlabel(
        "Predicted Grade"
    )

    ax.set_ylabel(
        "Actual Grade"
    )

    ax.set_xticks(
        np.arange(5)
    )

    ax.set_yticks(
        np.arange(5)
    )

    ax.set_xticklabels(
        classes
    )

    ax.set_yticklabels(
        classes
    )

    for i in range(5):

        for j in range(5):

            ax.text(
                j,
                i,
                f"{normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=11
            )

    plt.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "normalized_confusion_matrix_experiment5.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Normalized confusion matrix saved:\n"
        f"{output_path}"
    )


# ============================================================
# CLASS METRICS
# ============================================================

def calculate_class_metrics(cm):

    cm = np.array(
        cm,
        dtype=float
    )

    metrics = []

    for i in range(
        len(cm)
    ):

        tp = cm[i, i]

        fn = (
            cm[i, :].sum()
            - tp
        )

        fp = (
            cm[:, i].sum()
            - tp
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        if precision + recall > 0:

            f1 = (
                2 *
                precision *
                recall /
                (precision + recall)
            )

        else:

            f1 = 0

        metrics.append({

            "grade": i,

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            )
        })

    return metrics


# ============================================================
# CLASS PERFORMANCE CHART
# ============================================================

def create_class_performance_chart(
    class_metrics
):

    grades = [
        f"Grade {x['grade']}"
        for x in class_metrics
    ]

    precision = [
        x["precision"]
        for x in class_metrics
    ]

    recall = [
        x["recall"]
        for x in class_metrics
    ]

    f1 = [
        x["f1"]
        for x in class_metrics
    ]

    x = np.arange(
        len(grades)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bars1 = ax.bar(
        x - width,
        precision,
        width,
        label="Precision"
    )

    bars2 = ax.bar(
        x,
        recall,
        width,
        label="Recall"
    )

    bars3 = ax.bar(
        x + width,
        f1,
        width,
        label="F1 Score"
    )

    ax.set_title(
        "Experiment 5 - Per-Class Performance"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xlabel(
        "Osteoarthritis Grade"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        grades
    )

    ax.set_ylim(
        0,
        1
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bars in [
        bars1,
        bars2,
        bars3
    ]:

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "class_performance_experiment5.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Class performance chart saved:\n"
        f"{output_path}"
    )


# ============================================================
# SAVE ANALYSIS JSON
# ============================================================

def save_analysis(
    cm,
    class_metrics
):

    analysis = {

        "confusion_matrix":
            cm.tolist(),

        "class_metrics":
            class_metrics
    }

    output_path = (
        OUTPUT_DIR /
        "experiment5_class_analysis.json"
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            analysis,
            f,
            indent=4
        )

    print(
        f"Class analysis JSON saved:\n"
        f"{output_path}"
    )


# ============================================================
# PRINT ANALYSIS
# ============================================================

def print_analysis(
    class_metrics
):

    print("\n")

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 5 CLASS PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"{'Grade':<12}"
        f"{'Precision':>15}"
        f"{'Recall':>15}"
        f"{'F1 Score':>15}"
    )

    print(
        "-" * 57
    )

    for item in class_metrics:

        print(
            f"Grade {item['grade']:<5}"
            f"{item['precision']:>15.4f}"
            f"{item['recall']:>15.4f}"
            f"{item['f1']:>15.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nLoading Experiment 5 metrics..."
    )

    data = load_metrics()

    cm = find_confusion_matrix(
        data
    )

    if cm is None:

        raise ValueError(
            "Could not find confusion matrix "
            "inside experiment5_metrics.json."
        )

    cm = np.array(
        cm,
        dtype=int
    )

    if cm.shape != (5, 5):

        raise ValueError(
            f"Expected 5x5 confusion matrix, "
            f"got {cm.shape}"
        )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )

    # --------------------------------------------------------
    # CLASS METRICS
    # --------------------------------------------------------

    class_metrics = calculate_class_metrics(
        cm
    )

    print_analysis(
        class_metrics
    )

    # --------------------------------------------------------
    # CREATE VISUALIZATIONS
    # --------------------------------------------------------

    create_confusion_matrix(
        cm
    )

    create_normalized_confusion_matrix(
        cm
    )

    create_class_performance_chart(
        class_metrics
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    save_analysis(
        cm,
        class_metrics
    )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 5 ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    main()