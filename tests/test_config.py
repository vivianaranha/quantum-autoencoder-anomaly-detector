"""Configuration tests. Created by School of AI and School of QC."""

import json

import pytest

from quantum_autoencoder_anomaly_detector.config import ExperimentConfig


def test_default_config_is_valid():
    assert ExperimentConfig().validate().sample_count == 900


def test_json_round_trip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(ExperimentConfig().to_dict()))
    assert ExperimentConfig.from_json(path) == ExperimentConfig()


def test_overrides_ignore_none():
    assert ExperimentConfig().with_overrides(sample_count=None).sample_count == 900


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_count", 399),
        ("anomaly_rate", 0.01),
        ("validation_size", 0.05),
        ("test_size", 0.40),
        ("normal_train_limit", 39),
        ("input_qubits", 4),
        ("latent_qubits", 3),
        ("ansatz_reps", 0),
        ("optimizer_maxiter", 19),
        ("pca_latent_components", 8),
        ("false_negative_cost", 0),
        ("false_positive_cost", 6),
        ("output_root", ""),
    ],
)
def test_invalid_config_values(field, value):
    with pytest.raises(ValueError):
        ExperimentConfig().with_overrides(**{field: value})


def test_split_sum_is_validated():
    with pytest.raises(ValueError, match="cannot exceed"):
        ExperimentConfig(validation_size=0.31, test_size=0.31).validate()
