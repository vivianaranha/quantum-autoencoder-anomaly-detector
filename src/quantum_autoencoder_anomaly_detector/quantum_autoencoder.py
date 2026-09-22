"""Variational quantum autoencoder training and anomaly scoring.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .data import deterministic_normal_subset
from .models import ModelResult, finalize_external_result


def build_encoder_circuit(input_qubits: int, ansatz_reps: int) -> tuple[Any, Any]:
    """Build a hardware-efficient encoder with rotation and circular entangling layers."""

    if input_qubits < 2:
        raise ValueError("input_qubits must be at least 2")
    if ansatz_reps < 1:
        raise ValueError("ansatz_reps must be at least 1")

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    parameter_count = input_qubits * 2 * (ansatz_reps + 1)
    parameters = ParameterVector("theta", parameter_count)
    circuit = QuantumCircuit(input_qubits, name="QAE Encoder")
    position = 0
    for layer in range(ansatz_reps + 1):
        for qubit in range(input_qubits):
            circuit.ry(parameters[position], qubit)
            position += 1
            circuit.rz(parameters[position], qubit)
            position += 1
        if layer < ansatz_reps:
            for qubit in range(input_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            circuit.cx(input_qubits - 1, 0)
    return circuit, parameters


def _validated_amplitudes(amplitudes: np.ndarray, qubits: int) -> np.ndarray:
    values = np.asarray(amplitudes, dtype=complex)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 2**qubits or len(values) == 0:
        raise ValueError(f"amplitudes must have shape (samples, {2**qubits})")
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("amplitudes must be finite")
    probabilities = np.sum(np.abs(values) ** 2, axis=1)
    if not np.allclose(probabilities, 1.0, atol=1e-8):
        raise ValueError("every amplitude row must have unit norm")
    return values


def _bound_unitary(circuit: Any, parameters: Any, weights: np.ndarray) -> np.ndarray:
    from qiskit.quantum_info import Operator

    values = np.asarray(weights, dtype=float)
    if values.shape != (len(parameters),) or not np.isfinite(values).all():
        raise ValueError("weights must contain one finite value per circuit parameter")
    bindings = dict(zip(parameters, values, strict=True))
    return np.asarray(Operator(circuit.assign_parameters(bindings)).data, dtype=complex)


def encode_quantum_states(
    amplitudes: np.ndarray,
    circuit: Any,
    parameters: Any,
    weights: np.ndarray,
) -> np.ndarray:
    """Apply the learned encoder unitary to normalized input statevectors."""

    values = _validated_amplitudes(amplitudes, circuit.num_qubits)
    unitary = _bound_unitary(circuit, parameters, weights)
    return (unitary @ values.T).T


def compress_quantum_states(
    amplitudes: np.ndarray,
    circuit: Any,
    parameters: Any,
    weights: np.ndarray,
    latent_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the encoded state onto a zeroed trash register.

    Returns conditional latent statevectors and the probability that the trash
    register is all-zero. Qubits ``0..latent_qubits-1`` form the latent register.
    """

    if not 1 <= latent_qubits < circuit.num_qubits:
        raise ValueError("latent_qubits must be between 1 and input_qubits - 1")
    encoded = encode_quantum_states(amplitudes, circuit, parameters, weights)
    latent_dimension = 2**latent_qubits
    projected = encoded[:, :latent_dimension].copy()
    success_probability = np.sum(np.abs(projected) ** 2, axis=1).real
    nonzero = success_probability > 1e-15
    projected[nonzero] /= np.sqrt(success_probability[nonzero, None])
    projected[~nonzero] = 0.0
    projected[~nonzero, 0] = 1.0
    return projected, np.clip(success_probability, 0.0, 1.0)


def reconstruct_quantum_states(
    latent_states: np.ndarray,
    circuit: Any,
    parameters: Any,
    weights: np.ndarray,
    latent_qubits: int,
) -> np.ndarray:
    """Attach an all-zero trash register and apply the inverse encoder."""

    if not 1 <= latent_qubits < circuit.num_qubits:
        raise ValueError("latent_qubits must be between 1 and input_qubits - 1")
    latent = _validated_amplitudes(latent_states, latent_qubits)
    reference_states = np.zeros((len(latent), 2**circuit.num_qubits), dtype=complex)
    reference_states[:, : 2**latent_qubits] = latent
    unitary = _bound_unitary(circuit, parameters, weights)
    return (unitary.conj().T @ reference_states.T).T


def state_fidelity(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Return pure-state fidelity for aligned batches of normalized states."""

    first = np.asarray(original, dtype=complex)
    second = np.asarray(reconstructed, dtype=complex)
    if first.ndim == 1:
        first = first[None, :]
    if second.ndim == 1:
        second = second[None, :]
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("state batches must have the same two-dimensional shape")
    qubits = int(np.log2(first.shape[1]))
    if 2**qubits != first.shape[1]:
        raise ValueError("statevector width must be a power of two")
    first = _validated_amplitudes(first, qubits)
    second = _validated_amplitudes(second, qubits)
    overlap = np.sum(np.conjugate(first) * second, axis=1)
    return np.clip(np.abs(overlap) ** 2, 0.0, 1.0)


def trash_probability(
    amplitudes: np.ndarray,
    circuit: Any,
    parameters: Any,
    weights: np.ndarray,
    latent_qubits: int,
) -> np.ndarray:
    """Calculate one minus the reference-state fidelity of the trash register."""

    _, reference_probability = compress_quantum_states(
        amplitudes, circuit, parameters, weights, latent_qubits
    )
    return np.clip(1.0 - reference_probability, 0.0, 1.0)


@dataclass(slots=True)
class QuantumAutoencoderModel:
    """Serializable configuration and trained weights for quantum scoring."""

    input_qubits: int
    latent_qubits: int
    ansatz_reps: int
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.input_qubits < 2:
            raise ValueError("input_qubits must be at least 2")
        if not 1 <= self.latent_qubits < self.input_qubits:
            raise ValueError("latent_qubits must be between 1 and input_qubits - 1")
        if self.ansatz_reps < 1:
            raise ValueError("ansatz_reps must be at least 1")
        expected = self.input_qubits * 2 * (self.ansatz_reps + 1)
        values = np.asarray(self.weights, dtype=float)
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(f"weights must contain {expected} finite values")
        self.weights = values

    def score_samples(self, amplitudes: np.ndarray) -> np.ndarray:
        circuit, parameters = build_encoder_circuit(self.input_qubits, self.ansatz_reps)
        return trash_probability(amplitudes, circuit, parameters, self.weights, self.latent_qubits)

    def compress(self, amplitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return conditional latent states and compression-success probabilities."""

        circuit, parameters = build_encoder_circuit(self.input_qubits, self.ansatz_reps)
        return compress_quantum_states(
            amplitudes,
            circuit,
            parameters,
            self.weights,
            self.latent_qubits,
        )

    def reconstruct(self, amplitudes: np.ndarray) -> np.ndarray:
        """Compress and decode input states with a zeroed trash register."""

        circuit, parameters = build_encoder_circuit(self.input_qubits, self.ansatz_reps)
        latent, _ = compress_quantum_states(
            amplitudes,
            circuit,
            parameters,
            self.weights,
            self.latent_qubits,
        )
        return reconstruct_quantum_states(
            latent,
            circuit,
            parameters,
            self.weights,
            self.latent_qubits,
        )

    def reconstruction_fidelity(self, amplitudes: np.ndarray) -> np.ndarray:
        """Return round-trip pure-state fidelity after latent compression."""

        values = _validated_amplitudes(amplitudes, self.input_qubits)
        return state_fidelity(values, self.reconstruct(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_qubits": self.input_qubits,
            "latent_qubits": self.latent_qubits,
            "trash_qubits": self.input_qubits - self.latent_qubits,
            "ansatz_reps": self.ansatz_reps,
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> QuantumAutoencoderModel:
        return cls(
            input_qubits=int(values["input_qubits"]),
            latent_qubits=int(values["latent_qubits"]),
            ansatz_reps=int(values["ansatz_reps"]),
            weights=np.asarray(values["weights"], dtype=float),
        )


@dataclass(slots=True)
class QuantumTrainingOutput:
    """QAE result plus optimizer and training-subset provenance."""

    result: ModelResult
    model: QuantumAutoencoderModel
    circuit: Any
    objective_history: list[float]
    selected_ids: np.ndarray
    optimizer_info: dict[str, Any]


def train_quantum_autoencoder(
    amplitude_train: np.ndarray,
    train_ids: np.ndarray,
    amplitude_validation: np.ndarray,
    y_validation: np.ndarray,
    amplitude_test: np.ndarray,
    y_test: np.ndarray,
    normal_train_limit: int,
    input_qubits: int,
    latent_qubits: int,
    ansatz_reps: int,
    optimizer_maxiter: int,
    random_seed: int,
    false_negative_cost: float,
    false_positive_cost: float,
) -> QuantumTrainingOutput:
    """Train only on normal states and evaluate mixed validation/test telemetry."""

    selected_train, selected_ids, _ = deterministic_normal_subset(
        amplitude_train, train_ids, normal_train_limit, random_seed
    )
    circuit, parameters = build_encoder_circuit(input_qubits, ansatz_reps)
    rng = np.random.default_rng(random_seed)
    initial_point = rng.uniform(-0.1, 0.1, len(parameters))
    objective_history: list[float] = []

    def objective(weights: np.ndarray) -> float:
        value = float(
            np.mean(
                trash_probability(
                    selected_train,
                    circuit,
                    parameters,
                    weights,
                    latent_qubits,
                )
            )
        )
        objective_history.append(value)
        return value

    train_start = perf_counter()
    optimization = minimize(
        objective,
        initial_point,
        method="COBYLA",
        options={"maxiter": optimizer_maxiter, "rhobeg": 0.5, "tol": 1e-5},
    )
    training_seconds = perf_counter() - train_start
    model = QuantumAutoencoderModel(
        input_qubits=input_qubits,
        latent_qubits=latent_qubits,
        ansatz_reps=ansatz_reps,
        weights=np.asarray(optimization.x, dtype=float),
    )
    inference_start = perf_counter()
    raw_validation = model.score_samples(amplitude_validation)
    raw_test = model.score_samples(amplitude_test)
    inference_seconds = perf_counter() - inference_start
    result = finalize_external_result(
        name="quantum_autoencoder",
        model=model,
        raw_validation_scores=raw_validation,
        raw_test_scores=raw_test,
        validation_labels=y_validation,
        test_labels=y_test,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        training_samples=len(selected_train),
    )
    normal_mask = np.asarray(y_test, dtype=int) == 0
    anomaly_mask = ~normal_mask
    fidelities = 1.0 - raw_test
    result.metrics["mean_normal_reconstruction_fidelity"] = float(np.mean(fidelities[normal_mask]))
    result.metrics["mean_anomaly_reconstruction_fidelity"] = float(
        np.mean(fidelities[anomaly_mask])
    )
    result.metrics["reconstruction_fidelity_gap"] = float(
        result.metrics["mean_normal_reconstruction_fidelity"]
        - result.metrics["mean_anomaly_reconstruction_fidelity"]
    )
    optimizer_info = {
        "success": bool(optimization.success),
        "status": int(optimization.status),
        "message": str(optimization.message),
        "objective": float(optimization.fun),
        "evaluations": int(optimization.nfev),
    }
    return QuantumTrainingOutput(
        result=result,
        model=model,
        circuit=circuit,
        objective_history=objective_history,
        selected_ids=selected_ids,
        optimizer_info=optimizer_info,
    )


def circuit_diagnostics(
    circuit: Any,
    input_qubits: int,
    latent_qubits: int,
    ansatz_reps: int,
) -> dict[str, Any]:
    decomposed = circuit.decompose()
    return {
        "input_qubits": input_qubits,
        "latent_qubits": latent_qubits,
        "trash_qubits": input_qubits - latent_qubits,
        "ansatz_reps": ansatz_reps,
        "input_state_dimension": 2**input_qubits,
        "latent_state_dimension": 2**latent_qubits,
        "qubit_reduction_fraction": float((input_qubits - latent_qubits) / input_qubits),
        "state_dimension_reduction_fraction": float(1.0 - (2**latent_qubits / 2**input_qubits)),
        "latent_qubit_indices": list(range(latent_qubits)),
        "trash_qubit_indices": list(range(latent_qubits, input_qubits)),
        "trainable_parameters": int(circuit.num_parameters),
        "encoder_depth": int(decomposed.depth()),
        "encoder_operations": {
            str(name): int(count) for name, count in decomposed.count_ops().items()
        },
    }
