# ============================================================
# EXPERIMENT 5 - ERROR ANALYSIS
# ============================================================

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

METRICS_PATH = Path(
    "artifacts/metrics/experiment5_metrics.json"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics():

    print(
        "Loading Experiment 5 metrics..."
    )

    with open(
        METRICS_PATH,
        "r"
    ) as f:

        metrics = json.load(f)

    return metrics


# ============================================================
# EXTRACT CONFUSION MATRIX
# ============================================================

def get_confusion_matrix(metrics):

    confusion_matrix = metrics.get(
        "confusion_matrix"
    )

    if confusion_matrix is None:

        raise ValueError(
            "Confusion matrix not found "
            "in experiment5_metrics.json"
        )

    return np.array(
        confusion_matrix
    )


# ============================================================
# ERROR STATISTICS
# ============================================================

def calculate_error_statistics(
    confusion_matrix
):

    num_classes = confusion_matrix.shape[0]

    total_samples = confusion_matrix.sum()

    correct = np.trace(
        confusion_matrix
    )

    accuracy = (
        correct /
        total_samples
    )

    total_errors = (
        total_samples -
        correct
    )

    # --------------------------------------------------------
    # Absolute error
    # --------------------------------------------------------

    absolute_error_sum = 0

    severe_errors = 0

    adjacent_errors = 0

    for true_class in range(
        num_classes
    ):

        for predicted_class in range(
            num_classes
        ):

            count = confusion_matrix[
                true_class,
                predicted_class
            ]

            difference = abs(
                true_class -
                predicted_class
            )

            absolute_error_sum += (
                difference *
                count
            )

            if difference >= 2:

                severe_errors += count

            elif difference == 1:

                adjacent_errors += count

    mean_absolute_grade_error = (
        absolute_error_sum /
        total_samples
    )

    severe_error_rate = (
        severe_errors /
        total_samples
    )

    adjacent_error_rate = (
        adjacent_errors /
        total_samples
    )

    return {

        "total_samples":
            int(total_samples),

        "correct_predictions":
            int(correct),

        "incorrect_predictions":
            int(total_errors),

        "accuracy":
            float(accuracy),

        "mean_absolute_grade_error":
            float(
                mean_absolute_grade_error
            ),

        "adjacent_grade_errors":
            int(
                adjacent_errors
            ),

        "adjacent_error_rate":
            float(
                adjacent_error_rate
            ),

        "severe_errors":
            int(
                severe_errors
            ),

        "severe_error_rate":
            float(
                severe_error_rate
            )
    }


# ============================================================
# PER-CLASS ERROR ANALYSIS
# ============================================================

def calculate_class_errors(
    confusion_matrix
):

    num_classes = (
        confusion_matrix.shape[0]
    )

    results = {}

    for class_index in range(
        num_classes
    ):

        total = confusion_matrix[
            class_index
        ].sum()

        correct = confusion_matrix[
            class_index,
            class_index
        ]

        errors = (
            total -
            correct
        )

        recall = (
            correct / total
            if total > 0
            else 0
        )

        error_rate = (
            errors / total
            if total > 0
            else 0
        )

        results[str(class_index)] = {

            "total_samples":
                int(total),

            "correct":
                int(correct),

            "errors":
                int(errors),

            "recall":
                float(recall),

            "error_rate":
                float(error_rate)
        }

    return results


# ============================================================
# MOST COMMON MISCLASSIFICATIONS
# ============================================================

def get_common_errors(
    confusion_matrix
):

    num_classes = (
        confusion_matrix.shape[0]
    )

    errors = []

    for true_class in range(
        num_classes
    ):

        for predicted_class in range(
            num_classes
        ):

            if (
                true_class ==
                predicted_class
            ):
                continue

            count = confusion_matrix[
                true_class,
                predicted_class
            ]

            if count > 0:

                errors.append({

                    "true_grade":
                        true_class,

                    "predicted_grade":
                        predicted_class,

                    "count":
                        int(count),

                    "absolute_grade_error":
                        abs(
                            true_class -
                            predicted_class
                        )
                })

    errors.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    return errors


# ============================================================
# GRADE DISTANCE ANALYSIS
# ============================================================

def calculate_distance_distribution(
    confusion_matrix
):

    num_classes = (
        confusion_matrix.shape[0]
    )

    distribution = {}

    for distance in range(
        num_classes
    ):

        count = 0

        for true_class in range(
            num_classes
        ):

            for predicted_class in range(
                num_classes
            ):

                if abs(
                    true_class -
                    predicted_class
                ) == distance:

                    count += confusion_matrix[
                        true_class,
                        predicted_class
                    ]

        distribution[
            str(distance)
        ] = int(count)

    return distribution


# ============================================================
# SAVE ERROR SUMMARY
# ============================================================

def save_error_analysis(
    statistics,
    class_errors,
    common_errors,
    distance_distribution
):

    output = {

        "overall_error_statistics":
            statistics,

        "class_error_statistics":
            class_errors,

        "most_common_misclassifications":
            common_errors,

        "grade_distance_distribution":
            distance_distribution
    }

    output_path = (
        OUTPUT_DIR /
        "experiment5_error_analysis.json"
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    return output_path


# ============================================================
# PLOT GRADE ERROR DISTRIBUTION
# ============================================================

def plot_grade_distance(
    distance_distribution
):

    distances = [
        int(x)
        for x in
        distance_distribution.keys()
    ]

    counts = [
        distance_distribution[
            str(x)
        ]
        for x in distances
    ]

    labels = []

    for distance in distances:

        if distance == 0:

            labels.append(
                "Correct"
            )

        elif distance == 1:

            labels.append(
                "1 Grade"
            )

        else:

            labels.append(
                f"{distance} Grades"
            )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        counts
    )

    plt.xlabel(
        "Prediction Distance"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Experiment 5 Prediction Error Distance"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "grade_error_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    return output_path


# ============================================================
# PLOT PER-CLASS ERROR RATE
# ============================================================

def plot_class_error_rates(
    class_errors
):

    grades = [
        int(x)
        for x in class_errors.keys()
    ]

    error_rates = [
        class_errors[str(x)][
            "error_rate"
        ]
        for x in grades
    ]

    labels = [
        f"Grade {x}"
        for x in grades
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        error_rates
    )

    plt.xlabel(
        "True Grade"
    )

    plt.ylabel(
        "Error Rate"
    )

    plt.title(
        "Experiment 5 Per-Class Error Rate"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "class_error_rates.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    return output_path


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    statistics,
    class_errors,
    common_errors,
    distance_distribution
):

    print("\n")
    print("=" * 70)

    print(
        "EXPERIMENT 5 ERROR ANALYSIS"
    )

    print("=" * 70)

    print(
        f"Total Samples: "
        f"{statistics['total_samples']}"
    )

    print(
        f"Correct Predictions: "
        f"{statistics['correct_predictions']}"
    )

    print(
        f"Incorrect Predictions: "
        f"{statistics['incorrect_predictions']}"
    )

    print(
        f"Accuracy: "
        f"{statistics['accuracy']:.4f}"
    )

    print(
        f"Mean Absolute Grade Error: "
        f"{statistics['mean_absolute_grade_error']:.4f}"
    )

    print(
        f"Adjacent Grade Errors: "
        f"{statistics['adjacent_grade_errors']}"
    )

    print(
        f"Adjacent Error Rate: "
        f"{statistics['adjacent_error_rate']:.4f}"
    )

    print(
        f"Severe Errors (>=2 grades): "
        f"{statistics['severe_errors']}"
    )

    print(
        f"Severe Error Rate: "
        f"{statistics['severe_error_rate']:.4f}"
    )

    # --------------------------------------------------------
    # CLASS ERRORS
    # --------------------------------------------------------

    print("\n")
    print(
        "CLASS ERROR RATES"
    )

    print("-" * 70)

    print(
        f"{'Grade':<12}"
        f"{'Samples':<12}"
        f"{'Correct':<12}"
        f"{'Errors':<12}"
        f"{'Error Rate':<12}"
    )

    print("-" * 70)

    for grade, values in class_errors.items():

        print(
            f"{'Grade ' + grade:<12}"
            f"{values['total_samples']:<12}"
            f"{values['correct']:<12}"
            f"{values['errors']:<12}"
            f"{values['error_rate']:<12.4f}"
        )

    # --------------------------------------------------------
    # COMMON ERRORS
    # --------------------------------------------------------

    print("\n")
    print(
        "MOST COMMON MISCLASSIFICATIONS"
    )

    print("-" * 70)

    for error in common_errors[:10]:

        print(
            f"Grade "
            f"{error['true_grade']}"
            f" -> Grade "
            f"{error['predicted_grade']}"
            f": "
            f"{error['count']} samples "
            f"(distance="
            f"{error['absolute_grade_error']})"
        )

    # --------------------------------------------------------
    # DISTANCE DISTRIBUTION
    # --------------------------------------------------------

    print("\n")
    print(
        "GRADE DISTANCE DISTRIBUTION"
    )

    print("-" * 70)

    for distance, count in (
        distance_distribution.items()
    ):

        print(
            f"Distance {distance}: "
            f"{count} samples"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    metrics = load_metrics()

    confusion_matrix = (
        get_confusion_matrix(
            metrics
        )
    )

    statistics = (
        calculate_error_statistics(
            confusion_matrix
        )
    )

    class_errors = (
        calculate_class_errors(
            confusion_matrix
        )
    )

    common_errors = (
        get_common_errors(
            confusion_matrix
        )
    )

    distance_distribution = (
        calculate_distance_distribution(
            confusion_matrix
        )
    )

    json_path = (
        save_error_analysis(
            statistics,
            class_errors,
            common_errors,
            distance_distribution
        )
    )

    distance_plot = (
        plot_grade_distance(
            distance_distribution
        )
    )

    class_plot = (
        plot_class_error_rates(
            class_errors
        )
    )

    print_report(
        statistics,
        class_errors,
        common_errors,
        distance_distribution
    )

    print("\n")
    print("=" * 70)

    print(
        "ERROR ANALYSIS COMPLETE"
    )

    print("=" * 70)

    print(
        "\nJSON saved:"
    )

    print(
        json_path
    )

    print(
        "\nGrade distance chart saved:"
    )

    print(
        distance_plot
    )

    print(
        "\nClass error chart saved:"
    )

    print(
        class_plot
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    main()