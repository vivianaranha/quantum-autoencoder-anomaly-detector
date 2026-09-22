"""Validation-only thresholding and anomaly-detection metrics.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_search(
    labels: np.ndarray,
    scores: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    for threshold in np.linspace(0.0, 1.0, 201):
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        cost = false_negative_cost * fn + false_positive_cost * fp
        rows.append(
            {
                "threshold": float(threshold),
                "cost": float(cost),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["cost", "recall", "threshold"], ascending=[True, False, False]).iloc[
        0
    ]
    return float(best["threshold"]), frame


def _lift_at_fraction(labels: np.ndarray, scores: np.ndarray, fraction: float = 0.10) -> float:
    base_rate = float(np.mean(labels))
    if base_rate == 0.0:
        return math.nan
    count = max(1, int(math.ceil(len(labels) * fraction)))
    ranked = np.argsort(scores)[::-1][:count]
    return float(np.mean(labels[ranked]) / base_rate)


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
    training_seconds: float,
    inference_seconds: float,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))
    except ValueError:
        roc_auc = math.nan
        average_precision = math.nan
    specificity = float(tn / (tn + fp)) if tn + fp else math.nan
    total_cost = false_negative_cost * fn + false_positive_cost * fp
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "matthews_correlation": float(matthews_corrcoef(labels, predictions)),
        "lift_at_10_percent": _lift_at_fraction(labels, scores),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "business_cost": float(total_cost),
        "cost_per_1000": float(total_cost / len(labels) * 1_000),
        "flagged_samples": int(np.sum(predictions)),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
    }
