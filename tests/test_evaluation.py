"""Evaluation tests. Created by School of AI and School of QC."""

import numpy as np

from quantum_autoencoder_anomaly_detector.evaluation import (
    classification_metrics,
    threshold_search,
)


def test_threshold_search_finds_zero_cost_separator():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, history = threshold_search(labels, scores, 5.0, 1.0)
    best = history.loc[history.threshold == threshold].iloc[0]
    assert best.cost == 0
    assert 0.2 < threshold <= 0.8


def test_classification_metrics_perfect_case():
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5, 5, 1, 0.1, 0.01
    )
    assert metrics["f2"] == 1.0
    assert metrics["business_cost"] == 0.0
    assert metrics["true_positive"] == 2
