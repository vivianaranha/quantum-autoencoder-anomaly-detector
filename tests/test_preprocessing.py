"""Preprocessing tests. Created by School of AI and School of QC."""

import numpy as np

from quantum_autoencoder_anomaly_detector.data import FEATURE_COLUMNS, create_partitions
from quantum_autoencoder_anomaly_detector.preprocessing import amplitude_normalize, prepare_features


def test_amplitude_rows_have_unit_norm():
    values = np.arange(1, 25, dtype=float).reshape(3, 8)
    assert np.allclose(np.linalg.norm(amplitude_normalize(values), axis=1), 1.0)


def test_zero_row_maps_to_zero_basis_state():
    transformed = amplitude_normalize(np.zeros((1, 8)))
    assert np.array_equal(transformed[0], np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))


def test_prepare_features_shapes_and_train_mean():
    parts = create_partitions(400, 0.10, 0.20, 0.20, 42)
    prepared = prepare_features(parts.x_train, parts.x_validation, parts.x_test)
    assert prepared.standard_train.shape[1] == 8
    assert np.allclose(prepared.standard_train.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(prepared.amplitude_test, axis=1), 1.0)
    assert list(prepared.scaler.feature_names_in_) == FEATURE_COLUMNS
