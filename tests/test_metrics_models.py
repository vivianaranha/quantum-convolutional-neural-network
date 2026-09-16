"""Metric and model helper tests.

Created by School of AI and School of QC.
"""

import numpy as np
import pandas as pd
import pytest

from quantum_cnn.config import BenchmarkConfig
from quantum_cnn.data import make_image_dataset
from quantum_cnn.metrics import classification_metrics, summarize_metrics, winner_counts
from quantum_cnn.models import (
    FittedQCNN,
    linear_parameter_count,
    make_logistic_regression,
    make_mlp,
    make_rbf_svm,
    mlp_parameter_count,
    svm_support_vector_count,
)
from quantum_cnn.qcnn import QuantumConvolutionalNeuralNetwork


def test_perfect_metrics() -> None:
    metrics = classification_metrics(
        np.array([0, 1, 0, 1]),
        np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]),
        0.25,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_summary_uses_population_standard_deviation() -> None:
    rows = []
    for repeat, accuracy in enumerate((0.5, 1.0)):
        row = {"repeat": repeat, "model": "A"}
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "fit_seconds",
        ):
            row[metric] = accuracy
        rows.append(row)
    result = summarize_metrics(pd.DataFrame(rows))
    assert result["accuracy_mean"].iloc[0] == pytest.approx(0.75)
    assert result["accuracy_std"].iloc[0] == pytest.approx(0.25)


def test_winner_counts_include_ties() -> None:
    frame = pd.DataFrame(
        {
            "repeat": [0, 0, 0, 1, 1, 1],
            "model": ["Q", "S", "M", "Q", "S", "M"],
            "accuracy": [0.9, 0.9, 0.8, 0.7, 0.8, 0.9],
        }
    )
    counts = winner_counts(frame).set_index("model")["split_wins_or_ties"].to_dict()
    assert counts == {"Q": 1, "S": 1, "M": 1}


def test_winner_counts_handles_empty_frame() -> None:
    result = winner_counts(pd.DataFrame(columns=["repeat", "model", "accuracy"]))
    assert result.empty


def test_fitted_qcnn_round_trip() -> None:
    images = make_image_dataset(4, 0.1, 0.0, 2).images
    fitted = FittedQCNN(QuantumConvolutionalNeuralNetwork.initialize(3))
    restored = FittedQCNN.from_dict(fitted.to_dict())
    np.testing.assert_allclose(restored.predict_proba(images), fitted.predict_proba(images))


def test_classical_factories_fit_and_report_complexity() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(20, 8))
    labels = np.arange(20) % 2
    logistic = make_logistic_regression(1).fit(features, labels)
    svm = make_rbf_svm(1).fit(features, labels)
    mlp = make_mlp(BenchmarkConfig(mlp_max_iterations=100), 1).fit(features, labels)
    assert logistic.predict_proba(features).shape == (20, 2)
    assert svm.predict_proba(features).shape == (20, 2)
    assert mlp.predict_proba(features).shape == (20, 2)
    assert linear_parameter_count(logistic) == 9
    assert svm_support_vector_count(svm) > 0
    assert mlp_parameter_count(mlp) == 81
