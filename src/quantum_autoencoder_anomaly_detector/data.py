"""Synthetic industrial telemetry and clean-normal training partitions.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "temperature_c",
    "pressure_bar",
    "vibration_mm_s",
    "flow_l_min",
    "motor_current_a",
    "acoustic_db",
    "rotation_rpm",
    "efficiency_ratio",
]
ANOMALY_TYPES = ["sensor_spike", "process_drift", "correlation_break", "stuck_sensor"]


@dataclass(slots=True)
class DataPartitions:
    """Clean-normal training data and mixed validation/test data."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    excluded_train_anomaly_ids: np.ndarray
    full_dataset: pd.DataFrame


def generate_sensor_telemetry(
    sample_count: int,
    anomaly_rate: float,
    random_seed: int,
) -> pd.DataFrame:
    """Generate correlated operating telemetry with four injected anomaly families."""

    rng = np.random.default_rng(random_seed)
    load = rng.uniform(0.0, 1.0, sample_count)
    phase = rng.uniform(-1.0, 1.0, sample_count)
    base_vibration = 0.15 + 0.70 * load**2 + 0.08 * phase
    values = np.column_stack(
        [
            55.0 + 25.0 * load + 2.0 * phase + rng.normal(0.0, 1.0, sample_count),
            28.0 + 12.0 * load - 1.5 * phase + rng.normal(0.0, 0.6, sample_count),
            base_vibration + rng.normal(0.0, 0.025, sample_count),
            110.0 + 45.0 * load - 8.0 * base_vibration + rng.normal(0.0, 2.0, sample_count),
            7.0 + 5.0 * load + 0.8 * phase + rng.normal(0.0, 0.2, sample_count),
            38.0 + 10.0 * base_vibration + rng.normal(0.0, 0.7, sample_count),
            900.0 + 180.0 * load + rng.normal(0.0, 8.0, sample_count),
            0.70 + 0.20 * load - 0.08 * phase + rng.normal(0.0, 0.015, sample_count),
        ]
    )

    anomaly_count = max(1, int(round(sample_count * anomaly_rate)))
    anomaly_indices = rng.choice(sample_count, anomaly_count, replace=False)
    anomaly_names = np.full(sample_count, "normal", dtype=object)
    labels = np.zeros(sample_count, dtype=int)
    labels[anomaly_indices] = 1
    spike_scales = np.array([8.0, 5.0, 0.30, 18.0, 1.8, 5.0, 65.0, 0.10])

    for sequence, row_index in enumerate(anomaly_indices):
        anomaly_type = ANOMALY_TYPES[sequence % len(ANOMALY_TYPES)]
        anomaly_names[row_index] = anomaly_type
        if anomaly_type == "sensor_spike":
            feature_index = int(rng.integers(0, len(FEATURE_COLUMNS)))
            direction = rng.choice([-1.0, 1.0])
            values[row_index, feature_index] += (
                direction * spike_scales[feature_index] * rng.uniform(2.0, 3.5)
            )
        elif anomaly_type == "process_drift":
            values[row_index, 0] += rng.uniform(8.0, 14.0)
            values[row_index, 1] -= rng.uniform(4.0, 8.0)
            values[row_index, 4] += rng.uniform(1.5, 3.0)
        elif anomaly_type == "correlation_break":
            values[row_index, 0] += rng.uniform(-14.0, 14.0)
            values[row_index, 3] += rng.uniform(-30.0, 30.0)
            values[row_index, 6] += rng.uniform(-130.0, 130.0)
        else:
            values[row_index, 2] = rng.uniform(1.25, 1.55)
            values[row_index, 5] = rng.uniform(23.0, 29.0)

    start = np.datetime64("2026-01-01T00:00")
    timestamps = start + np.arange(sample_count).astype("timedelta64[m]")
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    for column in FEATURE_COLUMNS:
        frame[column] = frame[column].round(5)
    frame.insert(0, "timestamp", timestamps)
    frame.insert(0, "sample_id", [f"SENSOR-{index:06d}" for index in range(1, sample_count + 1)])
    frame["anomaly_type"] = anomaly_names
    frame["is_anomaly"] = labels
    return frame


def create_partitions(
    sample_count: int,
    anomaly_rate: float,
    validation_size: float,
    test_size: float,
    random_seed: int,
) -> DataPartitions:
    """Create a clean-normal train split and stratified mixed validation/test splits."""

    dataset = generate_sensor_telemetry(sample_count, anomaly_rate, random_seed)
    labels = dataset["is_anomaly"].to_numpy(dtype=int)
    positions = np.arange(sample_count)
    development_positions, test_positions = train_test_split(
        positions,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    training_pool, validation_positions = train_test_split(
        development_positions,
        test_size=relative_validation_size,
        random_state=random_seed,
        stratify=labels[development_positions],
    )
    clean_training_positions = training_pool[labels[training_pool] == 0]
    excluded_positions = training_pool[labels[training_pool] == 1]

    def features(selected: np.ndarray) -> pd.DataFrame:
        return dataset.loc[selected, FEATURE_COLUMNS].copy()

    def identifiers(selected: np.ndarray) -> np.ndarray:
        return dataset.loc[selected, "sample_id"].to_numpy(dtype=str)

    return DataPartitions(
        x_train=features(clean_training_positions),
        x_validation=features(validation_positions),
        x_test=features(test_positions),
        y_train=labels[clean_training_positions].copy(),
        y_validation=labels[validation_positions].copy(),
        y_test=labels[test_positions].copy(),
        train_ids=identifiers(clean_training_positions),
        validation_ids=identifiers(validation_positions),
        test_ids=identifiers(test_positions),
        excluded_train_anomaly_ids=identifiers(excluded_positions),
        full_dataset=dataset,
    )


def deterministic_normal_subset(
    x_train: np.ndarray,
    train_ids: np.ndarray,
    limit: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a simulator-friendly normal training subset."""

    rng = np.random.default_rng(random_seed)
    count = min(limit, len(x_train))
    positions = rng.choice(len(x_train), size=count, replace=False)
    return x_train[positions], train_ids[positions], positions


def export_sensor_telemetry(
    path: str | Path,
    sample_count: int,
    anomaly_rate: float,
    random_seed: int,
) -> None:
    generate_sensor_telemetry(sample_count, anomaly_rate, random_seed).to_csv(path, index=False)
