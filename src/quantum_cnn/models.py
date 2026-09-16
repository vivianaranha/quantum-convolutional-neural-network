"""Matched-input classical baselines and fitted QCNN wrapper.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import BenchmarkConfig
from .data import extract_patch_features
from .qcnn import QuantumConvolutionalNeuralNetwork


@dataclass(slots=True)
class FittedQCNN:
    model: QuantumConvolutionalNeuralNetwork

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(extract_patch_features(images))

    def predict(self, images: np.ndarray) -> np.ndarray:
        return (self.predict_proba(images)[:, 1] >= 0.5).astype(int)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "preprocessing": {
                "image_shape": [8, 8],
                "patch_grid": [4, 2],
                "patch_shape": [2, 4],
                "feature_order": "row-major patch means",
                "clip_range": [0.0, 1.0],
            },
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FittedQCNN:
        return cls(QuantumConvolutionalNeuralNetwork.from_dict(values["model"]))


def make_logistic_regression(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=1.0, max_iter=2_000, random_state=random_seed),
            ),
        ]
    )


def make_rbf_svm(random_seed: int) -> Pipeline:
    del random_seed  # The deterministic SVC/calibration path has no stochastic state.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    SVC(C=3.0, kernel="rbf", gamma="scale"),
                    cv=3,
                    ensemble=False,
                ),
            ),
        ]
    )


def make_mlp(config: BenchmarkConfig, random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(config.mlp_hidden_units,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=0.001,
                    max_iter=config.mlp_max_iterations,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def linear_parameter_count(model: Pipeline) -> int:
    estimator = model.named_steps["model"]
    return int(estimator.coef_.size + estimator.intercept_.size)


def mlp_parameter_count(model: Pipeline) -> int:
    estimator = model.named_steps["model"]
    return int(
        sum(weights.size for weights in estimator.coefs_)
        + sum(bias.size for bias in estimator.intercepts_)
    )


def svm_support_vector_count(model: Pipeline) -> int:
    calibrator = model.named_steps["model"]
    estimator = calibrator.calibrated_classifiers_[0].estimator
    return int(estimator.support_vectors_.shape[0])
