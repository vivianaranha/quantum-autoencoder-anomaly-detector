"""Model tests. Created by School of AI and School of QC."""

import numpy as np

from quantum_autoencoder_anomaly_detector.data import create_partitions
from quantum_autoencoder_anomaly_detector.models import ScoreNormalizer, train_classical_models
from quantum_autoencoder_anomaly_detector.preprocessing import prepare_features


def test_score_normalizer_clips_and_scales():
    scaler = ScoreNormalizer.fit(np.array([2.0, 4.0]))
    assert np.array_equal(scaler.transform(np.array([1.0, 3.0, 5.0])), [0.0, 0.5, 1.0])


def test_constant_score_normalizer_returns_zero():
    scaler = ScoreNormalizer.fit(np.array([2.0, 2.0]))
    assert np.array_equal(scaler.transform(np.array([2.0, 3.0])), [0.0, 0.0])


def test_classical_models_share_result_contract():
    parts = create_partitions(400, 0.10, 0.20, 0.20, 42)
    prepared = prepare_features(parts.x_train, parts.x_validation, parts.x_test)
    results = train_classical_models(
        prepared.standard_train,
        prepared.standard_validation,
        parts.y_validation,
        prepared.standard_test,
        parts.y_test,
        2,
        42,
        5.0,
        1.0,
    )
    assert set(results) == {"pca_autoencoder", "isolation_forest"}
    for result in results.values():
        assert len(result.raw_test_scores) == len(parts.y_test)
        assert len(result.raw_validation_scores) == len(parts.y_validation)
        assert len(result.predictions) == len(parts.y_test)
        assert 0 <= result.threshold <= 1
        assert 0 <= result.metrics["roc_auc"] <= 1
