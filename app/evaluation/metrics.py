"""Pure metric calculations for the evaluation harness. No I/O and no project imports.

Conventions (also explained in every report):
  - A rate whose denominator is zero is None ("n/a"), never 0.0. Precision is undefined when
    nothing was predicted positive; recall is undefined when there are no actual positives.
  - F1 is None when either precision or recall is None; otherwise 2*TP / (2*TP + FP + FN).
  - Macro averages cover only classes that have at least one actual example. A class with examples
    that is never predicted contributes 0 (and is listed), so missing a class lowers the score.
  - 95% intervals use the Wilson score method, which behaves sensibly for small samples.
"""

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

Z_95 = 1.959963984540054
NO_PREDICTION = "no_prediction"


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None else safe_ratio(2 * tp, 2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion, or None when there is no data."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    if total == 0:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _ci(interval: tuple[float, float] | None) -> list[float] | None:
    return None if interval is None else [interval[0], interval[1]]


def binary_report(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": tp + fn,
        "predicted_positive": tp + fp,
        "units": tp + fp + fn + tn,
        **precision_recall_f1(tp, fp, fn),
        "precision_ci95": _ci(wilson_interval(tp, tp + fp)),
        "recall_ci95": _ci(wilson_interval(tp, tp + fn)),
    }


def micro_average(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    reports = list(reports)
    return binary_report(
        sum(r["tp"] for r in reports),
        sum(r["fp"] for r in reports),
        sum(r["fn"] for r in reports),
        sum(r["tn"] for r in reports),
    )


def macro_average(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    included = [name for name, report in reports.items() if report["support"] > 0]
    if not included:
        return {"classes": [], "zero_division_classes": [], "precision": None, "recall": None, "f1": None}
    zero_division = [name for name in included if reports[name]["precision"] is None]
    return {
        "classes": included,
        "zero_division_classes": zero_division,
        "precision": statistics.fmean(reports[n]["precision"] or 0.0 for n in included),
        "recall": statistics.fmean(reports[n]["recall"] or 0.0 for n in included),
        "f1": statistics.fmean(reports[n]["f1"] or 0.0 for n in included),
    }


def multiclass_report(
    labels: Sequence[str],
    pairs: Iterable[tuple[str, str]],
    *,
    extra_predictions: Sequence[str] = (NO_PREDICTION,),
) -> dict[str, Any]:
    """Confusion matrix and per-class metrics for (actual, predicted) pairs.

    `extra_predictions` are answers that are not classes (by default "no_prediction": no valid
    answer was produced). They appear as extra matrix columns and count as errors for the actual class.
    """
    labels = list(labels)
    predicted_columns = labels + [extra for extra in extra_predictions if extra not in labels]
    matrix = {actual: {predicted: 0 for predicted in predicted_columns} for actual in labels}
    n = 0
    for actual, predicted in pairs:
        if actual not in matrix:
            raise ValueError(f"unknown actual label: {actual!r}")
        if predicted not in predicted_columns:
            raise ValueError(f"unknown predicted label: {predicted!r}")
        matrix[actual][predicted] += 1
        n += 1

    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = matrix[label][label]
        fn = sum(count for predicted, count in matrix[label].items() if predicted != label)
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        per_class[label] = binary_report(tp, fp, fn, n - tp - fn - fp)

    correct = sum(matrix[label][label] for label in labels)
    supports = {label: per_class[label]["support"] for label in labels}
    majority = max(labels, key=lambda label: supports[label]) if labels else None
    return {
        "labels": labels,
        "predicted_columns": predicted_columns,
        "matrix": matrix,
        "n": n,
        "correct": correct,
        "accuracy": safe_ratio(correct, n),
        "accuracy_ci95": _ci(wilson_interval(correct, n)),
        "no_prediction_count": sum(row.get(NO_PREDICTION, 0) for row in matrix.values()),
        "majority_class": majority,
        "majority_baseline_accuracy": safe_ratio(supports[majority], n) if majority else None,
        "per_class": per_class,
        "macro": macro_average(per_class),
        "micro": micro_average(per_class.values()),
    }


def summarize_latencies(samples_ms: Sequence[float]) -> dict[str, Any]:
    if not samples_ms:
        return {"n": 0, "min_ms": None, "median_ms": None, "max_ms": None, "mean_ms": None}
    return {
        "n": len(samples_ms),
        "min_ms": round(min(samples_ms), 3),
        "median_ms": round(statistics.median(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
        "mean_ms": round(statistics.fmean(samples_ms), 3),
    }