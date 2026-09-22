"""Validated experiment configuration.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Controls telemetry, autoencoders, thresholds, costs, and outputs."""

    sample_count: int = 900
    anomaly_rate: float = 0.12
    validation_size: float = 0.20
    test_size: float = 0.20
    random_seed: int = 42
    normal_train_limit: int = 250
    input_qubits: int = 3
    latent_qubits: int = 1
    ansatz_reps: int = 2
    optimizer_maxiter: int = 100
    pca_latent_components: int = 2
    false_negative_cost: float = 5.0
    false_positive_cost: float = 1.0
    include_quantum_autoencoder: bool = True
    output_root: str = "artifacts"

    def validate(self) -> ExperimentConfig:
        if not 400 <= self.sample_count <= 10_000:
            raise ValueError("sample_count must be between 400 and 10,000")
        if not 0.05 <= self.anomaly_rate <= 0.30:
            raise ValueError("anomaly_rate must be between 0.05 and 0.30")
        if not 0.10 <= self.validation_size <= 0.35:
            raise ValueError("validation_size must be between 0.10 and 0.35")
        if not 0.10 <= self.test_size <= 0.35:
            raise ValueError("test_size must be between 0.10 and 0.35")
        if self.validation_size + self.test_size > 0.60:
            raise ValueError("validation_size plus test_size cannot exceed 0.60")
        if not 40 <= self.normal_train_limit <= 1_000:
            raise ValueError("normal_train_limit must be between 40 and 1,000")
        if self.input_qubits != 3:
            raise ValueError("input_qubits must be 3 for the eight-feature amplitude encoding")
        if not 1 <= self.latent_qubits < self.input_qubits:
            raise ValueError("latent_qubits must be at least 1 and smaller than input_qubits")
        if not 1 <= self.ansatz_reps <= 4:
            raise ValueError("ansatz_reps must be between 1 and 4")
        parameter_count = self.input_qubits * 2 * (self.ansatz_reps + 1)
        if not parameter_count + 2 <= self.optimizer_maxiter <= 500:
            raise ValueError(f"optimizer_maxiter must be between {parameter_count + 2} and 500")
        if not 1 <= self.pca_latent_components < 8:
            raise ValueError("pca_latent_components must be between 1 and 7")
        if self.false_negative_cost <= 0 or self.false_positive_cost <= 0:
            raise ValueError("misclassification costs must be positive")
        if self.false_negative_cost < self.false_positive_cost:
            raise ValueError("false_negative_cost must be at least false_positive_cost")
        if not self.output_root.strip():
            raise ValueError("output_root cannot be empty")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        values = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **values).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        return cls(**values).validate()
