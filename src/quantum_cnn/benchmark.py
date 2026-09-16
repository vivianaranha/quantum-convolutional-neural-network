"""End-to-end QCNN image-classification benchmark.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from .artifacts import create_run_directory, save_artifacts
from .config import BenchmarkConfig
from .data import ImageDataset, extract_patch_features, make_image_dataset
from .metrics import classification_metrics, summarize_metrics, winner_counts
from .models import (
    FittedQCNN,
    linear_parameter_count,
    make_logistic_regression,
    make_mlp,
    make_rbf_svm,
    mlp_parameter_count,
    svm_support_vector_count,
)
from .qcnn import QuantumConvolutionalNeuralNetwork, circuit_diagnostics, sigmoid, train_qcnn

QCNN_NAME = "Quantum Convolutional Neural Network"
LOGISTIC_NAME = "Logistic Regression"
SVM_NAME = "RBF SVM"
MLP_NAME = "Neural Network"


@dataclass(slots=True)
class BenchmarkResult:
    output_directory: Path
    summary: pd.DataFrame
    per_split: pd.DataFrame
    predictions: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    robustness: pd.DataFrame
    training_history: pd.DataFrame
    winner_counts: pd.DataFrame
    circuit_info: dict[str, object]


def _prediction_rows(
    dataset: ImageDataset,
    repeat: int,
    test_indices: np.ndarray,
    model_name: str,
    probabilities: np.ndarray,
) -> list[dict[str, float | int | bool | str]]:
    return [
        {
            "repeat": repeat,
            "row_id": int(row_id),
            "model": model_name,
            "label": int(dataset.labels[row_id]),
            "occluded": bool(dataset.occluded[row_id]),
            "probability_horizontal": float(probability[1]),
            "prediction": int(probability[1] >= 0.5),
        }
        for row_id, probability in zip(test_indices, probabilities, strict=True)
    ]


def _split_rows(
    repeat: int,
    split_seed: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for split, indices in (("train", train_indices), ("test", test_indices)):
        rows.extend(
            {
                "repeat": repeat,
                "split_seed": split_seed,
                "row_id": int(index),
                "split": split,
            }
            for index in indices
        )
    return rows


def _robustness_rows(
    model: QuantumConvolutionalNeuralNetwork,
    features: np.ndarray,
    labels: np.ndarray,
    repeat: int,
    config: BenchmarkConfig,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(config.random_seed + 30_000 + repeat)
    exact_probability = model.predict_proba(features)[:, 1]
    clean_accuracy = float(np.mean((exact_probability >= 0.5) == labels))
    rows: list[dict[str, float | int | str]] = []

    perturbed_accuracies = []
    for _ in range(config.noise_trials):
        perturbed = QuantumConvolutionalNeuralNetwork.from_dict(model.to_dict())
        perturbed.weights += rng.normal(0.0, config.parameter_noise, size=perturbed.weights.shape)
        perturbed_accuracies.append(float(np.mean(perturbed.predict(features) == labels)))
    rows.append(
        {
            "repeat": repeat,
            "probe": "parameter_noise",
            "setting": config.parameter_noise,
            "trials": config.noise_trials,
            "clean_accuracy": clean_accuracy,
            "perturbed_accuracy_mean": float(np.mean(perturbed_accuracies)),
            "perturbed_accuracy_std": float(np.std(perturbed_accuracies, ddof=0)),
            "accuracy_change": float(np.mean(perturbed_accuracies) - clean_accuracy),
        }
    )

    expectation = model.expectation(features)
    probability_one = np.clip((1.0 - expectation) / 2.0, 0.0, 1.0)
    shot_accuracies = []
    for _ in range(config.shot_trials):
        counts = rng.binomial(config.shot_count, probability_one)
        estimated_expectation = 1.0 - 2.0 * counts / config.shot_count
        probability = sigmoid(model.head_scale * estimated_expectation + model.head_bias)
        shot_accuracies.append(float(np.mean((probability >= 0.5) == labels)))
    rows.append(
        {
            "repeat": repeat,
            "probe": "finite_shots",
            "setting": config.shot_count,
            "trials": config.shot_trials,
            "clean_accuracy": clean_accuracy,
            "perturbed_accuracy_mean": float(np.mean(shot_accuracies)),
            "perturbed_accuracy_std": float(np.std(shot_accuracies, ddof=0)),
            "accuracy_change": float(np.mean(shot_accuracies) - clean_accuracy),
        }
    )
    return rows


def _subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    for (repeat, model, occluded), frame in predictions.groupby(
        ["repeat", "model", "occluded"], sort=False
    ):
        labels = frame["label"].to_numpy(dtype=int)
        predicted = frame["prediction"].to_numpy(dtype=int)
        rows.append(
            {
                "repeat": int(repeat),
                "model": str(model),
                "occluded": bool(occluded),
                "samples": len(frame),
                "accuracy": float(np.mean(predicted == labels)),
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(
    config: BenchmarkConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    resolved = (config or BenchmarkConfig()).validate()

    def notify(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    dataset = make_image_dataset(
        resolved.samples,
        resolved.image_noise,
        resolved.occlusion_probability,
        resolved.random_seed,
    )
    features = extract_patch_features(dataset.images)
    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | int | bool | str]] = []
    split_rows: list[dict[str, int | str]] = []
    history_frames: list[pd.DataFrame] = []
    robustness_rows: list[dict[str, float | int | str]] = []
    first_models: dict[str, object] = {}

    row_indices = np.arange(resolved.samples)
    for repeat in range(resolved.repeats):
        notify(f"Training repeat {repeat + 1} of {resolved.repeats}")
        split_seed = resolved.random_seed + resolved.split_seed_offset + repeat
        train_indices, test_indices = train_test_split(
            row_indices,
            test_size=resolved.test_size,
            stratify=dataset.labels,
            random_state=split_seed,
        )
        split_rows.extend(_split_rows(repeat, split_seed, train_indices, test_indices))
        x_train, x_test = features[train_indices], features[test_indices]
        y_train, y_test = dataset.labels[train_indices], dataset.labels[test_indices]
        model_seed = resolved.random_seed + 20_000 + repeat

        start = perf_counter()
        trained = train_qcnn(
            x_train,
            y_train,
            resolved.qcnn_epochs,
            resolved.qcnn_learning_rate,
            resolved.qcnn_l2,
            model_seed,
        )
        fit_seconds = perf_counter() - start
        qcnn_probability = trained.model.predict_proba(x_test)
        metric_rows.append(
            {
                "repeat": repeat,
                "split_seed": split_seed,
                "model": QCNN_NAME,
                "complexity_value": trained.model.parameter_count,
                "complexity_unit": "trainable_parameters",
                **classification_metrics(y_test, qcnn_probability, fit_seconds),
            }
        )
        prediction_rows.extend(
            _prediction_rows(dataset, repeat, test_indices, QCNN_NAME, qcnn_probability)
        )
        history = trained.history.copy()
        history.insert(0, "repeat", repeat)
        history_frames.append(history)
        robustness_rows.extend(_robustness_rows(trained.model, x_test, y_test, repeat, resolved))

        logistic = make_logistic_regression(model_seed)
        start = perf_counter()
        logistic.fit(x_train, y_train)
        logistic_seconds = perf_counter() - start
        logistic_probability = logistic.predict_proba(x_test)
        metric_rows.append(
            {
                "repeat": repeat,
                "split_seed": split_seed,
                "model": LOGISTIC_NAME,
                "complexity_value": linear_parameter_count(logistic),
                "complexity_unit": "trainable_parameters",
                **classification_metrics(y_test, logistic_probability, logistic_seconds),
            }
        )
        prediction_rows.extend(
            _prediction_rows(dataset, repeat, test_indices, LOGISTIC_NAME, logistic_probability)
        )

        svm = make_rbf_svm(model_seed)
        start = perf_counter()
        svm.fit(x_train, y_train)
        svm_seconds = perf_counter() - start
        svm_probability = svm.predict_proba(x_test)
        metric_rows.append(
            {
                "repeat": repeat,
                "split_seed": split_seed,
                "model": SVM_NAME,
                "complexity_value": svm_support_vector_count(svm),
                "complexity_unit": "support_vectors",
                **classification_metrics(y_test, svm_probability, svm_seconds),
            }
        )
        prediction_rows.extend(
            _prediction_rows(dataset, repeat, test_indices, SVM_NAME, svm_probability)
        )

        mlp = make_mlp(resolved, model_seed)
        start = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp.fit(x_train, y_train)
        mlp_seconds = perf_counter() - start
        mlp_probability = mlp.predict_proba(x_test)
        metric_rows.append(
            {
                "repeat": repeat,
                "split_seed": split_seed,
                "model": MLP_NAME,
                "complexity_value": mlp_parameter_count(mlp),
                "complexity_unit": "trainable_parameters",
                **classification_metrics(y_test, mlp_probability, mlp_seconds),
            }
        )
        prediction_rows.extend(
            _prediction_rows(dataset, repeat, test_indices, MLP_NAME, mlp_probability)
        )

        if repeat == 0:
            first_models = {
                QCNN_NAME: FittedQCNN(trained.model),
                LOGISTIC_NAME: logistic,
                SVM_NAME: svm,
                MLP_NAME: mlp,
            }

    per_split = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    splits = pd.DataFrame(split_rows)
    training_history = pd.concat(history_frames, ignore_index=True)
    robustness = pd.DataFrame(robustness_rows)
    summary = summarize_metrics(per_split)
    wins = winner_counts(per_split)
    subgroup = _subgroup_metrics(predictions)
    circuit_info = circuit_diagnostics(resolved.qcnn_epochs)
    destination = create_run_directory(resolved.output_root)
    notify("Saving models, metrics, images, diagnostics, and plots")
    save_artifacts(
        destination,
        resolved,
        dataset,
        features,
        first_models,
        per_split,
        summary,
        predictions,
        splits,
        subgroup,
        training_history,
        robustness,
        wins,
        circuit_info,
    )
    notify(f"Complete: {destination}")
    return BenchmarkResult(
        destination,
        summary,
        per_split,
        predictions,
        subgroup,
        robustness,
        training_history,
        wins,
        circuit_info,
    )
