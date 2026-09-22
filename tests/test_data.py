"""Data tests. Created by School of AI and School of QC."""

import pandas as pd

from quantum_autoencoder_anomaly_detector.data import (
    ANOMALY_TYPES,
    FEATURE_COLUMNS,
    create_partitions,
    deterministic_normal_subset,
    generate_sensor_telemetry,
)


def test_generator_schema_and_exact_prevalence():
    frame = generate_sensor_telemetry(900, 0.12, 42)
    assert frame.shape == (900, 12)
    assert frame["is_anomaly"].sum() == 108
    assert set(FEATURE_COLUMNS).issubset(frame.columns)


def test_four_anomaly_families_are_balanced():
    frame = generate_sensor_telemetry(900, 0.12, 42)
    counts = frame.loc[frame.is_anomaly == 1, "anomaly_type"].value_counts()
    assert set(counts.index) == set(ANOMALY_TYPES)
    assert counts.nunique() == 1


def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(
        generate_sensor_telemetry(400, 0.10, 7),
        generate_sensor_telemetry(400, 0.10, 7),
    )


def test_partition_sizes_and_clean_training():
    parts = create_partitions(500, 0.10, 0.20, 0.20, 42)
    assert len(parts.y_validation) == 100
    assert len(parts.y_test) == 100
    assert len(parts.y_train) == 270
    assert parts.y_train.sum() == 0
    assert len(parts.excluded_train_anomaly_ids) == 30


def test_partition_ids_cover_dataset_once():
    parts = create_partitions(500, 0.10, 0.20, 0.20, 42)
    combined = list(parts.train_ids) + list(parts.validation_ids) + list(parts.test_ids)
    combined += list(parts.excluded_train_anomaly_ids)
    assert len(combined) == len(set(combined)) == 500


def test_deterministic_subset_respects_limit():
    parts = create_partitions(400, 0.10, 0.20, 0.20, 42)
    values = parts.x_train.to_numpy()
    subset, identifiers, positions = deterministic_normal_subset(values, parts.train_ids, 40, 9)
    assert subset.shape == (40, 8)
    assert len(identifiers) == len(positions) == 40
