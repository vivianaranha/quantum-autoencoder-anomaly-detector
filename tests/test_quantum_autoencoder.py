"""Quantum autoencoder tests. Created by School of AI and School of QC."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from quantum_autoencoder_anomaly_detector.data import create_partitions
from quantum_autoencoder_anomaly_detector.preprocessing import prepare_features
from quantum_autoencoder_anomaly_detector.quantum_autoencoder import (
    QuantumAutoencoderModel,
    build_encoder_circuit,
    circuit_diagnostics,
    compress_quantum_states,
    encode_quantum_states,
    reconstruct_quantum_states,
    state_fidelity,
    train_quantum_autoencoder,
    trash_probability,
)


def test_encoder_parameter_count_and_diagnostics():
    circuit, parameters = build_encoder_circuit(3, 2)
    diagnostics = circuit_diagnostics(circuit, 3, 1, 2)
    assert len(parameters) == diagnostics["trainable_parameters"] == 18
    assert diagnostics["trash_qubits"] == 2
    assert diagnostics["encoder_depth"] > 0


def test_trash_scores_are_probabilities():
    circuit, parameters = build_encoder_circuit(3, 1)
    amplitudes = np.eye(8)
    scores = trash_probability(amplitudes, circuit, parameters, np.zeros(12), 1)
    assert scores.shape == (8,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_quantum_model_serialization_round_trip():
    original = QuantumAutoencoderModel(3, 1, 1, np.arange(12) / 10)
    restored = QuantumAutoencoderModel.from_dict(original.to_dict())
    assert restored.input_qubits == 3
    assert np.array_equal(restored.weights, original.weights)


def test_small_quantum_training_run():
    parts = create_partitions(400, 0.10, 0.20, 0.20, 42)
    prepared = prepare_features(parts.x_train, parts.x_validation, parts.x_test)
    output = train_quantum_autoencoder(
        prepared.amplitude_train,
        parts.train_ids,
        prepared.amplitude_validation,
        parts.y_validation,
        prepared.amplitude_test,
        parts.y_test,
        40,
        3,
        1,
        1,
        14,
        42,
        5.0,
        1.0,
    )
    assert len(output.selected_ids) == 40
    assert output.optimizer_info["evaluations"] <= 14
    assert len(output.result.predictions) == len(parts.y_test)
    assert output.result.metrics["mean_normal_reconstruction_fidelity"] >= 0
    assert output.result.metrics["mean_anomaly_reconstruction_fidelity"] >= 0


def _random_states(seed, samples=4):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(samples, 8)) + 1j * rng.normal(size=(samples, 8))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


@pytest.mark.parametrize(("repetitions", "latent_qubits"), [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_trash_probability_matches_independent_qiskit_statevector(repetitions, latent_qubits):
    circuit, parameters = build_encoder_circuit(3, repetitions)
    rng = np.random.default_rng(100 + repetitions + latent_qubits)
    weights = rng.normal(scale=0.3, size=len(parameters))
    amplitudes = _random_states(200 + repetitions + latent_qubits)
    actual = trash_probability(amplitudes, circuit, parameters, weights, latent_qubits)
    bound = circuit.assign_parameters(dict(zip(parameters, weights, strict=True)))
    trash_qubits = list(range(latent_qubits, 3))
    expected = []
    for amplitudes_row in amplitudes:
        encoded = Statevector(amplitudes_row).evolve(bound)
        zero_trash_probability = encoded.probabilities(qargs=trash_qubits)[0]
        expected.append(1.0 - zero_trash_probability)
    assert np.allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("latent_qubits", [1, 2])
def test_compression_and_decoder_fidelity_equal_success_probability(latent_qubits):
    circuit, parameters = build_encoder_circuit(3, 2)
    weights = np.linspace(-0.4, 0.5, len(parameters))
    amplitudes = _random_states(91 + latent_qubits)
    latent, success = compress_quantum_states(
        amplitudes, circuit, parameters, weights, latent_qubits
    )
    reconstructed = reconstruct_quantum_states(latent, circuit, parameters, weights, latent_qubits)
    assert latent.shape == (len(amplitudes), 2**latent_qubits)
    assert np.allclose(np.linalg.norm(latent, axis=1), 1.0)
    assert np.allclose(state_fidelity(amplitudes, reconstructed), success, atol=1e-12)


def test_zero_weight_encoder_preserves_basis_state_and_perfect_fidelity():
    circuit, parameters = build_encoder_circuit(3, 1)
    input_state = np.zeros(8)
    input_state[0] = 1.0
    weights = np.zeros(len(parameters))
    encoded = encode_quantum_states(input_state, circuit, parameters, weights)
    model = QuantumAutoencoderModel(3, 1, 1, weights)
    assert np.allclose(encoded[0], input_state)
    assert model.score_samples(input_state)[0] == pytest.approx(0.0)
    assert model.reconstruction_fidelity(input_state)[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("qubits", "repetitions"),
    [(1, 1), (3, 0)],
)
def test_invalid_encoder_configuration_is_rejected(qubits, repetitions):
    with pytest.raises(ValueError):
        build_encoder_circuit(qubits, repetitions)


@pytest.mark.parametrize(
    "amplitudes",
    [
        np.ones((2, 7)),
        np.ones((2, 8)),
        np.full((2, 8), np.nan),
        np.empty((0, 8)),
    ],
)
def test_invalid_amplitudes_are_rejected(amplitudes):
    model = QuantumAutoencoderModel(3, 1, 1, np.zeros(12))
    with pytest.raises(ValueError):
        model.score_samples(amplitudes)


@pytest.mark.parametrize(
    ("input_qubits", "latent_qubits", "repetitions", "weights"),
    [
        (1, 1, 1, np.zeros(8)),
        (3, 0, 1, np.zeros(12)),
        (3, 3, 1, np.zeros(12)),
        (3, 1, 0, np.zeros(6)),
        (3, 1, 1, np.zeros(11)),
        (3, 1, 1, np.full(12, np.nan)),
    ],
)
def test_invalid_serializable_model_is_rejected(input_qubits, latent_qubits, repetitions, weights):
    with pytest.raises(ValueError):
        QuantumAutoencoderModel(input_qubits, latent_qubits, repetitions, weights)


def test_state_fidelity_rejects_mismatched_or_non_power_of_two_batches():
    with pytest.raises(ValueError, match="same"):
        state_fidelity(np.eye(8)[:2], np.eye(8)[:3])
    values = np.ones((2, 6)) / np.sqrt(6)
    with pytest.raises(ValueError, match="power of two"):
        state_fidelity(values, values)
