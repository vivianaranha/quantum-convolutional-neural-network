"""Static benchmark visualizations.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmark import LOGISTIC_NAME, MLP_NAME, QCNN_NAME, SVM_NAME
from .data import ImageDataset

MODEL_COLORS = {
    QCNN_NAME: "#6750A4",
    LOGISTIC_NAME: "#00639B",
    SVM_NAME: "#D97706",
    MLP_NAME: "#2E7D32",
}


def _finish(figure: plt.Figure, destination: Path) -> None:
    figure.tight_layout()
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_image_gallery(dataset: ImageDataset, destination: Path) -> None:
    figure, axes = plt.subplots(2, 6, figsize=(10.5, 4.2))
    selections = []
    for label in (0, 1):
        candidates = np.flatnonzero(dataset.labels == label)
        selections.extend(candidates[:6])
    for axis, row_id in zip(axes.ravel(), selections, strict=True):
        axis.imshow(dataset.images[row_id], cmap="magma", vmin=0, vmax=1)
        label = "Horizontal" if dataset.labels[row_id] else "Vertical"
        suffix = " · occluded" if dataset.occluded[row_id] else ""
        axis.set_title(f"{label}{suffix}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Generated 8×8 line-orientation images")
    _finish(figure, destination)


def plot_patch_features(dataset: ImageDataset, features: np.ndarray, destination: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(9.5, 4.5))
    for column, row_id in enumerate((0, 1)):
        axes[0, 2 * column].imshow(dataset.images[row_id], cmap="magma", vmin=0, vmax=1)
        axes[0, 2 * column].set_title("8×8 image")
        axes[0, 2 * column + 1].imshow(
            features[row_id].reshape(4, 2), cmap="viridis", vmin=0, vmax=1
        )
        axes[0, 2 * column + 1].set_title("4×2 patch means")
    means = [features[dataset.labels == label].mean(axis=0).reshape(4, 2) for label in (0, 1)]
    for column, (label, mean) in enumerate(zip(("Vertical", "Horizontal"), means, strict=True)):
        axis = axes[1, 2 * column : 2 * column + 2]
        axis[0].imshow(mean, cmap="viridis", vmin=0, vmax=1)
        axis[0].set_title(f"Mean {label.lower()} features")
        axis[1].bar(np.arange(8), mean.ravel(), color="#6750A4")
        axis[1].set(title=f"{label} qubit inputs", xlabel="Qubit", ylim=(0, 1))
    for axis in axes.ravel():
        if len(axis.images):
            axis.set_xticks([])
            axis.set_yticks([])
        axis.grid(alpha=0.15)
    figure.suptitle("Spatial compression before quantum encoding")
    _finish(figure, destination)


def plot_metric(summary: pd.DataFrame, metric: str, title: str, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    values = summary[f"{metric}_mean"]
    errors = summary[f"{metric}_std"]
    colors = [MODEL_COLORS[name] for name in summary["model"]]
    bars = axis.bar(summary["model"], values, yerr=errors, capsize=4, color=colors, alpha=0.9)
    axis.bar_label(bars, fmt="%.3f", padding=4)
    axis.set(title=title, ylabel=f"Mean {metric.replace('_', ' ')}", ylim=(0, 1.08))
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.2)
    _finish(figure, destination)


def plot_training(history: pd.DataFrame, destination: Path) -> None:
    grouped = history.groupby("epoch", sort=False).agg(
        loss=("training_loss", "mean"),
        loss_std=("training_loss", lambda values: float(np.std(values, ddof=0))),
        accuracy=("training_accuracy", "mean"),
        accuracy_std=("training_accuracy", lambda values: float(np.std(values, ddof=0))),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    epoch = grouped.index.to_numpy()
    axes[0].plot(epoch, grouped["loss"], color="#6750A4")
    axes[0].fill_between(
        epoch,
        grouped["loss"] - grouped["loss_std"],
        grouped["loss"] + grouped["loss_std"],
        color="#6750A4",
        alpha=0.18,
    )
    axes[1].plot(epoch, grouped["accuracy"], color="#00639B")
    axes[1].fill_between(
        epoch,
        grouped["accuracy"] - grouped["accuracy_std"],
        grouped["accuracy"] + grouped["accuracy_std"],
        color="#00639B",
        alpha=0.18,
    )
    axes[0].set(title="QCNN training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="QCNN training accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    for axis in axes:
        axis.grid(alpha=0.2)
    _finish(figure, destination)


def plot_robustness(robustness: pd.DataFrame, destination: Path) -> None:
    grouped = robustness.groupby("probe", sort=False).agg(
        clean=("clean_accuracy", "mean"),
        perturbed=("perturbed_accuracy_mean", "mean"),
        spread=("perturbed_accuracy_mean", lambda values: float(np.std(values, ddof=0))),
    )
    labels = [
        "Parameter noise" if name == "parameter_noise" else "Finite shots" for name in grouped.index
    ]
    positions = np.arange(len(grouped))
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(positions - 0.18, grouped["clean"], 0.36, label="Exact", color="#6750A4")
    axis.bar(
        positions + 0.18,
        grouped["perturbed"],
        0.36,
        yerr=grouped["spread"],
        capsize=4,
        label="Probe mean",
        color="#D97706",
    )
    axis.set_xticks(positions, labels)
    axis.set(title="QCNN sensitivity probes", ylabel="Held-out accuracy", ylim=(0, 1.05))
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    _finish(figure, destination)


def plot_fit_time(summary: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    colors = [MODEL_COLORS[name] for name in summary["model"]]
    bars = axis.bar(summary["model"], summary["fit_seconds_mean"], color=colors)
    axis.bar_label(bars, fmt="%.3fs", padding=3)
    axis.set_yscale("log")
    axis.set(title="Observed fit time (log scale)", ylabel="Mean wall-clock seconds")
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.2)
    _finish(figure, destination)


def plot_misclassifications(
    dataset: ImageDataset, predictions: pd.DataFrame, destination: Path
) -> None:
    first = predictions.loc[predictions["repeat"] == 0]
    model_names = list(MODEL_COLORS)
    figure, axes = plt.subplots(len(model_names), 6, figsize=(10, 7.4), squeeze=False)
    for row, model_name in enumerate(model_names):
        frame = first.loc[(first["model"] == model_name) & (first["label"] != first["prediction"])]
        row_ids = frame["row_id"].head(6).to_numpy(dtype=int)
        for column, axis in enumerate(axes[row]):
            if column < len(row_ids):
                row_id = row_ids[column]
                axis.imshow(dataset.images[row_id], cmap="magma", vmin=0, vmax=1)
                axis.set_title(
                    f"true {dataset.labels[row_id]} · pred {int(frame.iloc[column]['prediction'])}",
                    fontsize=7,
                )
            else:
                axis.text(0.5, 0.5, "No additional\nerror", ha="center", va="center", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(model_name, fontsize=8)
    figure.suptitle("First-repeat held-out misclassifications")
    _finish(figure, destination)


def save_all_plots(
    destination: Path,
    dataset: ImageDataset,
    features: np.ndarray,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    training_history: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    plot_image_gallery(dataset, destination / "image_gallery.png")
    plot_patch_features(dataset, features, destination / "patch_features.png")
    plot_metric(
        summary,
        "accuracy",
        "Held-out accuracy across repeated splits",
        destination / "accuracy.png",
    )
    plot_metric(summary, "f1", "Held-out F1 across repeated splits", destination / "f1.png")
    plot_training(training_history, destination / "qcnn_training.png")
    plot_robustness(robustness, destination / "qcnn_robustness.png")
    plot_fit_time(summary, destination / "fit_time.png")
    plot_misclassifications(dataset, predictions, destination / "misclassifications.png")
