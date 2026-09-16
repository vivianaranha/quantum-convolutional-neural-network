"""QCNN statevector, gradient, serialization, and circuit tests.

Created by School of AI and School of QC.
"""

import re

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from quantum_cnn.qcnn import (
    STAGE_PAIRS,
    QuantumConvolutionalNeuralNetwork,
    build_symbolic_circuit,
    circuit_diagnostics,
    encode_features,
    final_expectation,
    run_qcnn_states,
    sigmoid,
    train_qcnn,
)


def test_encoding_shape_and_normalization() -> None:
    encoded = encode_features(np.array([[0.1] * 8, [0.8] * 8]))
    assert encoded.shape == (2, 256)
    np.testing.assert_allclose(np.sum(np.abs(encoded) ** 2, axis=1), 1.0)


def test_encoding_accepts_single_row_and_clips() -> None:
    encoded = encode_features(np.array([-1.0, 2.0] * 4))
    assert encoded.shape == (1, 256)
    np.testing.assert_allclose(np.sum(np.abs(encoded) ** 2), 1.0)


def test_encoding_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        encode_features(np.zeros((2, 7)))


def test_qcnn_preserves_state_norm() -> None:
    encoded = encode_features(np.full((3, 8), 0.4))
    weights = np.linspace(-0.2, 0.3, 12).reshape(3, 4)
    states = run_qcnn_states(encoded, weights)
    np.testing.assert_allclose(np.sum(np.abs(states) ** 2, axis=1), 1.0, atol=1e-10)


def test_qcnn_validates_shapes() -> None:
    with pytest.raises(ValueError, match="256"):
        run_qcnn_states(np.zeros((2, 16)), np.zeros((3, 4)))
    with pytest.raises(ValueError, match="weights"):
        run_qcnn_states(np.zeros((2, 256)), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="256"):
        final_expectation(np.zeros((2, 16)))


def test_expectation_is_bounded() -> None:
    model = QuantumConvolutionalNeuralNetwork.initialize(3)
    values = model.expectation(np.full((4, 8), 0.5))
    assert ((values >= -1) & (values <= 1)).all()


def test_probabilities_and_serialization_round_trip() -> None:
    model = QuantumConvolutionalNeuralNetwork.initialize(5)
    features = np.array([[0.2] * 8, [0.7] * 8])
    probability = model.predict_proba(features)
    restored = QuantumConvolutionalNeuralNetwork.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.predict_proba(features), probability)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert model.parameter_count == 14


def test_sigmoid_is_stable() -> None:
    values = sigmoid(np.array([-1_000.0, 0.0, 1_000.0]))
    assert np.isfinite(values).all()
    assert values[0] < 1e-10
    assert values[1] == 0.5
    assert values[2] > 1 - 1e-10


def test_shared_parameter_shift_matches_finite_difference() -> None:
    rng = np.random.default_rng(8)
    encoded = encode_features(rng.random((2, 8)))
    weights = rng.normal(0.0, 0.2, size=(3, 4))
    stage, component = 0, 0
    shifted = np.zeros(2)
    for occurrence in range(len(STAGE_PAIRS[stage])):
        base = weights[stage, component]
        plus = final_expectation(
            run_qcnn_states(encoded, weights, (stage, component, occurrence, base + np.pi / 2))
        )
        minus = final_expectation(
            run_qcnn_states(encoded, weights, (stage, component, occurrence, base - np.pi / 2))
        )
        shifted += 0.5 * (plus - minus)
    epsilon = 1e-6
    plus_weights = weights.copy()
    minus_weights = weights.copy()
    plus_weights[stage, component] += epsilon
    minus_weights[stage, component] -= epsilon
    finite = (
        final_expectation(run_qcnn_states(encoded, plus_weights))
        - final_expectation(run_qcnn_states(encoded, minus_weights))
    ) / (2 * epsilon)
    np.testing.assert_allclose(shifted, finite, atol=1e-6)


def test_numpy_simulator_matches_qiskit() -> None:
    rng = np.random.default_rng(5)
    features = rng.random(8)
    weights = rng.normal(0.0, 0.2, size=(3, 4))
    circuit = build_symbolic_circuit()
    binding = {}
    for parameter in circuit.parameters:
        index = int(re.search(r"\[(\d+)\]", parameter.name).group(1))
        binding[parameter] = (
            features[index] if parameter.name.startswith("x") else weights.ravel()[index]
        )
    qiskit_state = np.asarray(Statevector.from_instruction(circuit.assign_parameters(binding)).data)
    numpy_state = run_qcnn_states(encode_features(features), weights)[0]
    np.testing.assert_allclose(numpy_state, qiskit_state, atol=1e-10)


def test_tiny_training_reduces_loss() -> None:
    features = np.vstack([np.full((8, 8), 0.15), np.full((8, 8), 0.85)])
    labels = np.array([0] * 8 + [1] * 8)
    result = train_qcnn(features, labels, 5, 0.04, 0.0, 9)
    assert result.history["training_loss"].iloc[-1] < result.history["training_loss"].iloc[0]
    assert np.isfinite(result.history.select_dtypes(include="number")).all().all()


def test_training_validates_inputs() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        train_qcnn(np.zeros((3, 7)), np.zeros(3), 5, 0.1, 0.0, 1)
    with pytest.raises(ValueError, match="binary"):
        train_qcnn(np.zeros((3, 8)), np.array([0, 1, 2]), 5, 0.1, 0.0, 1)
    with pytest.raises(ValueError, match="positive"):
        train_qcnn(np.zeros((3, 8)), np.array([0, 1, 0]), 0, 0.1, 0.0, 1)


def test_circuit_diagnostics_are_auditable() -> None:
    values = circuit_diagnostics(80)
    assert values["logical_qubits"] == 8
    assert values["active_qubits_by_stage"] == [8, 4, 2, 1]
    assert values["total_trainable_parameters"] == 14
    assert values["parameterized_gate_occurrences"] == 28
    assert values["shots"] is None
    assert values["statevector_passes_per_fit"] == 4_560
