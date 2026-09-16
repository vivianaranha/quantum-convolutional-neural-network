"""Classification metrics and repeated-holdout summaries.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
    "fit_seconds",
)


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, fit_seconds: float
) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    positive = values[:, 1]
    predictions = (positive >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, positive)),
        "log_loss": float(log_loss(y_true, values, labels=[0, 1])),
        "fit_seconds": float(fit_seconds),
    }


def summarize_metrics(per_split: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str | object]] = {}
    for metric in METRIC_COLUMNS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (
            metric,
            lambda values: float(np.std(values, ddof=0)),
        )
    return per_split.groupby("model", sort=False).agg(**aggregations).reset_index()


def winner_counts(per_split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for repeat, frame in per_split.groupby("repeat", sort=False):
        best = float(frame["accuracy"].max())
        for model in frame.loc[np.isclose(frame["accuracy"], best), "model"]:
            rows.append({"repeat": int(repeat), "model": model})
    if not rows:
        return pd.DataFrame(columns=["model", "split_wins_or_ties"])
    return (
        pd.DataFrame(rows)
        .groupby("model", sort=False)
        .size()
        .rename("split_wins_or_ties")
        .reset_index()
    )
