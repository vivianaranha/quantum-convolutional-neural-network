"""Eight-qubit quantum convolutional neural network and exact simulator.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

QUBITS = 8
STAGE_PAIRS = (
    ((0, 1), (2, 3), (4, 5), (6, 7)),
    ((1, 3), (5, 7)),
    ((3, 7),),
)


def _ry(angle: float) -> np.ndarray:
    cosine = np.cos(angle / 2)
    sine = np.sin(angle / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def _apply_single(states: np.ndarray, gate: np.ndarray, qubit: int) -> None:
    step = 1 << qubit
    view = states.reshape(len(states), -1, 2 * step)
    amplitude_zero = view[:, :, :step].copy()
    amplitude_one = view[:, :, step:].copy()
    view[:, :, :step] = gate[0, 0] * amplitude_zero + gate[0, 1] * amplitude_one
    view[:, :, step:] = gate[1, 0] * amplitude_zero + gate[1, 1] * amplitude_one


def _apply_cx(states: np.ndarray, control: int, target: int) -> None:
    if control == target:
        raise ValueError("control and target must differ")
    indices = np.arange(states.shape[1])
    selected = indices[((indices >> control) & 1 == 1) & ((indices >> target) & 1 == 0)]
    partner = selected | (1 << target)
    temporary = states[:, selected].copy()
    states[:, selected] = states[:, partner]
    states[:, partner] = temporary


def encode_features(features: np.ndarray) -> np.ndarray:
    """Angle-encode eight normalized patch means as product states."""

    values = np.asarray(features, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != QUBITS:
        raise ValueError("features must have shape (samples, 8)")
    clipped = np.clip(values, 0.0, 1.0)
    states = np.ones((len(clipped), 1), dtype=complex)
    for qubit in range(QUBITS):
        ry_angle = np.pi * clipped[:, qubit]
        rz_angle = np.pi * clipped[:, qubit] ** 2
        local = np.column_stack(
            [
                np.exp(-0.5j * rz_angle) * np.cos(0.5 * ry_angle),
                np.exp(0.5j * rz_angle) * np.sin(0.5 * ry_angle),
            ]
        )
        states = np.einsum("ni,nj->nij", local, states).reshape(len(clipped), -1)
    return states


def _gate_angle(
    weights: np.ndarray,
    stage: int,
    component: int,
    occurrence: int,
    override: tuple[int, int, int, float] | None,
) -> float:
    if override is not None and override[:3] == (stage, component, occurrence):
        return override[3]
    return float(weights[stage, component])


def run_qcnn_states(
    encoded: np.ndarray,
    weights: np.ndarray,
    override: tuple[int, int, int, float] | None = None,
) -> np.ndarray:
    """Apply three shared convolution/pooling stages to encoded states."""

    states = np.asarray(encoded, dtype=complex).copy()
    if states.ndim != 2 or states.shape[1] != 2**QUBITS:
        raise ValueError("encoded states must have shape (samples, 256)")
    parameters = np.asarray(weights, dtype=float)
    if parameters.shape != (3, 4):
        raise ValueError("weights must have shape (3, 4)")
    for stage, pairs in enumerate(STAGE_PAIRS):
        for occurrence, (source, sink) in enumerate(pairs):
            first = _gate_angle(parameters, stage, 0, occurrence, override)
            second = _gate_angle(parameters, stage, 1, occurrence, override)
            phase = _gate_angle(parameters, stage, 2, occurrence, override)
            _apply_single(states, _ry(first), source)
            _apply_single(states, _ry(second), sink)
            _apply_cx(states, source, sink)
            _apply_single(states, _rz(phase), sink)
            _apply_cx(states, sink, source)
        for occurrence, (source, sink) in enumerate(pairs):
            pooling = _gate_angle(parameters, stage, 3, occurrence, override)
            _apply_cx(states, source, sink)
            _apply_single(states, _ry(pooling), sink)
    return states


def final_expectation(states: np.ndarray) -> np.ndarray:
    """Return the exact Pauli-Z expectation on the retained qubit seven."""

    values = np.asarray(states, dtype=complex)
    if values.ndim != 2 or values.shape[1] != 2**QUBITS:
        raise ValueError("states must have shape (samples, 256)")
    indices = np.arange(values.shape[1])
    probabilities = np.abs(values) ** 2
    probability_one = probabilities[:, ((indices >> 7) & 1) == 1].sum(axis=1)
    return 1.0 - 2.0 * probability_one


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(slots=True)
class QuantumConvolutionalNeuralNetwork:
    weights: np.ndarray
    head_scale: float
    head_bias: float

    @classmethod
    def initialize(cls, random_seed: int) -> QuantumConvolutionalNeuralNetwork:
        rng = np.random.default_rng(random_seed)
        return cls(rng.normal(0.0, 0.22, size=(3, 4)), 1.0, 0.0)

    @property
    def quantum_parameter_count(self) -> int:
        return int(self.weights.size)

    @property
    def parameter_count(self) -> int:
        return self.quantum_parameter_count + 2

    def expectation(self, features: np.ndarray) -> np.ndarray:
        return final_expectation(run_qcnn_states(encode_features(features), self.weights))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = sigmoid(self.head_scale * self.expectation(features) + self.head_bias)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "eight_qubit_shared_qcnn",
            "weights": self.weights.tolist(),
            "head_scale": self.head_scale,
            "head_bias": self.head_bias,
            "quantum_parameter_count": self.quantum_parameter_count,
            "parameter_count": self.parameter_count,
            "active_qubits_by_stage": [8, 4, 2, 1],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> QuantumConvolutionalNeuralNetwork:
        return cls(
            np.asarray(values["weights"], dtype=float),
            float(values["head_scale"]),
            float(values["head_bias"]),
        )


class AdamDescent:
    def __init__(self, shapes: list[tuple[int, ...]], learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.moments = [np.zeros(shape, dtype=float) for shape in shapes]
        self.velocities = [np.zeros(shape, dtype=float) for shape in shapes]
        self.step = 0

    def update(self, parameters: list[np.ndarray], gradients: list[np.ndarray]) -> None:
        self.step += 1
        beta1, beta2 = 0.9, 0.999
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
            bounded = np.clip(gradient, -5.0, 5.0)
            self.moments[index] = beta1 * self.moments[index] + (1 - beta1) * bounded
            self.velocities[index] = beta2 * self.velocities[index] + (1 - beta2) * bounded**2
            moment = self.moments[index] / (1 - beta1**self.step)
            velocity = self.velocities[index] / (1 - beta2**self.step)
            parameter -= self.learning_rate * moment / (np.sqrt(velocity) + 1e-8)


@dataclass(slots=True)
class QCNNTrainingResult:
    model: QuantumConvolutionalNeuralNetwork
    history: pd.DataFrame


def train_qcnn(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2_penalty: float,
    random_seed: int,
) -> QCNNTrainingResult:
    """Train shared QCNN filters with exact occurrence-wise parameter shift."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[1] != QUBITS or len(x) != len(y):
        raise ValueError("features and labels have incompatible shapes")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be binary values 0 and 1")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model = QuantumConvolutionalNeuralNetwork.initialize(random_seed)
    encoded = encode_features(x)
    head_scale = np.array([model.head_scale], dtype=float)
    head_bias = np.array([model.head_bias], dtype=float)
    optimizer = AdamDescent([model.weights.shape, head_scale.shape, head_bias.shape], learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        expectation = final_expectation(run_qcnn_states(encoded, model.weights))
        probability = sigmoid(head_scale[0] * expectation + head_bias[0])
        error = probability - y
        loss = -np.mean(y * np.log(probability + 1e-12) + (1 - y) * np.log(1 - probability + 1e-12))
        loss += 0.5 * l2_penalty * float(np.sum(model.weights**2) + head_scale[0] ** 2)
        quantum_gradient = np.zeros_like(model.weights)
        for stage, pairs in enumerate(STAGE_PAIRS):
            for component in range(4):
                derivative = np.zeros(len(x), dtype=float)
                for occurrence in range(len(pairs)):
                    base = float(model.weights[stage, component])
                    plus = final_expectation(
                        run_qcnn_states(
                            encoded,
                            model.weights,
                            (stage, component, occurrence, base + np.pi / 2),
                        )
                    )
                    minus = final_expectation(
                        run_qcnn_states(
                            encoded,
                            model.weights,
                            (stage, component, occurrence, base - np.pi / 2),
                        )
                    )
                    derivative += 0.5 * (plus - minus)
                quantum_gradient[stage, component] = float(
                    np.mean(error * head_scale[0] * derivative)
                    + l2_penalty * model.weights[stage, component]
                )
        scale_gradient = np.array(
            [float(np.mean(error * expectation) + l2_penalty * head_scale[0])]
        )
        bias_gradient = np.array([float(np.mean(error))])
        optimizer.update(
            [model.weights, head_scale, head_bias],
            [quantum_gradient, scale_gradient, bias_gradient],
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(loss),
                "training_accuracy": float(np.mean((probability >= 0.5) == y)),
                "quantum_gradient_norm": float(np.linalg.norm(quantum_gradient)),
                "head_gradient_norm": float(
                    np.sqrt(scale_gradient[0] ** 2 + bias_gradient[0] ** 2)
                ),
            }
        )
    model.head_scale = float(head_scale[0])
    model.head_bias = float(head_bias[0])
    return QCNNTrainingResult(model, pd.DataFrame(history))


def build_symbolic_circuit() -> Any:
    """Build the feature-map and QCNN topology for visualization and diagnostics."""

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    features = ParameterVector("x", QUBITS)
    parameters = ParameterVector("theta", 12)
    circuit = QuantumCircuit(QUBITS, name="QCNN")
    for qubit in range(QUBITS):
        circuit.ry(np.pi * features[qubit], qubit)
        circuit.rz(np.pi * features[qubit] ** 2, qubit)
    for stage, pairs in enumerate(STAGE_PAIRS):
        offset = 4 * stage
        for source, sink in pairs:
            circuit.ry(parameters[offset], source)
            circuit.ry(parameters[offset + 1], sink)
            circuit.cx(source, sink)
            circuit.rz(parameters[offset + 2], sink)
            circuit.cx(sink, source)
        for source, sink in pairs:
            circuit.cx(source, sink)
            circuit.ry(parameters[offset + 3], sink)
        if stage < 2:
            circuit.barrier()
    return circuit


def circuit_diagnostics(epochs: int) -> dict[str, Any]:
    circuit = build_symbolic_circuit()
    occurrence_count = sum(len(pairs) * 4 for pairs in STAGE_PAIRS)
    return {
        "logical_qubits": QUBITS,
        "statevector_dimension": 2**QUBITS,
        "input_features": QUBITS,
        "active_qubits_by_stage": [8, 4, 2, 1],
        "convolution_stages": 3,
        "shared_quantum_parameters": 12,
        "classical_head_parameters": 2,
        "total_trainable_parameters": 14,
        "parameterized_gate_occurrences": occurrence_count,
        "circuit_depth": int(circuit.decompose().depth()),
        "operations": {str(name): int(count) for name, count in circuit.count_ops().items()},
        "execution": "exact NumPy statevector with Qiskit circuit parity",
        "shots": None,
        "statevector_passes_per_epoch": 1 + 2 * occurrence_count,
        "statevector_passes_per_fit": epochs * (1 + 2 * occurrence_count),
    }
