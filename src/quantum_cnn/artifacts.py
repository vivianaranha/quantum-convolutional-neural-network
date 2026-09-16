"""Artifact, model, report, and provenance persistence.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .data import ImageDataset
from .models import FittedQCNN
from .qcnn import build_symbolic_circuit

ATTRIBUTION = "Created by School of AI and School of QC"
QCNN_NAME = "Quantum Convolutional Neural Network"


def create_run_directory(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    destination = root / timestamp
    suffix = 1
    while destination.exists():
        destination = root / f"{timestamp}-{suffix}"
        suffix += 1
    destination.mkdir()
    return destination


def write_json(path: Path, values: Any) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "qiskit", "streamlit"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _report(
    config: BenchmarkConfig,
    summary: pd.DataFrame,
    wins: pd.DataFrame,
    robustness: pd.DataFrame,
    circuit_info: dict[str, Any],
) -> str:
    lines = [
        "# QCNN Benchmark Report",
        "",
        f"{ATTRIBUTION}.",
        "",
        "## Protocol",
        "",
        f"- Images: `{config.samples}` synthetic 8×8 line-orientation samples",
        f"- Occlusion probability: `{config.occlusion_probability:.2f}`",
        f"- Repeated stratified holdouts: `{config.repeats}`",
        f"- Test fraction: `{config.test_size:.2f}`",
        f"- QCNN epochs per split: `{config.qcnn_epochs}`",
        f"- Qubits: `{circuit_info['logical_qubits']}`",
        f"- QCNN trainable parameters: `{circuit_info['total_trainable_parameters']}`",
        "",
        "## Overall model results",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"- {row['model']}: accuracy `{row['accuracy_mean']:.4f} ± "
            f"{row['accuracy_std']:.4f}`, F1 `{row['f1_mean']:.4f}`, ROC AUC "
            f"`{row['roc_auc_mean']:.4f}`, fit time `{row['fit_seconds_mean']:.4f}s`."
        )
    lines.extend(["", "## Split wins or ties", ""])
    for row in wins.to_dict(orient="records"):
        lines.append(f"- {row['model']}: `{row['split_wins_or_ties']}`")
    lines.extend(["", "## QCNN sensitivity probes", ""])
    for probe, frame in robustness.groupby("probe", sort=False):
        lines.append(
            f"- {probe}: clean accuracy `{frame['clean_accuracy'].mean():.4f}`, "
            f"probe accuracy `{frame['perturbed_accuracy_mean'].mean():.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The QCNN uses exact statevector simulation. Finite-shot and parameter-noise "
            "results are controlled sensitivity probes, not physical-device experiments. "
            "The data is synthetic, the spatial compression is fixed, and the benchmark "
            "does not demonstrate quantum advantage.",
            "",
            f"{ATTRIBUTION}.",
            "",
        ]
    )
    return "\n".join(lines)


def save_artifacts(
    destination: Path,
    config: BenchmarkConfig,
    dataset: ImageDataset,
    features: np.ndarray,
    first_models: dict[str, object],
    per_split: pd.DataFrame,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    subgroup: pd.DataFrame,
    training_history: pd.DataFrame,
    robustness: pd.DataFrame,
    wins: pd.DataFrame,
    circuit_info: dict[str, Any],
) -> None:
    from .visualization import save_all_plots

    write_json(destination / "config.json", config.to_dict())
    write_json(destination / "dependency_versions.json", dependency_versions())
    write_json(destination / "circuit_diagnostics.json", circuit_info)
    write_json(
        destination / "metrics.json",
        {
            "attribution": ATTRIBUTION,
            "overall_summary": summary.to_dict(orient="records"),
            "split_wins_or_ties": wins.to_dict(orient="records"),
        },
    )
    dataset.to_frame().to_csv(destination / "generated_images.csv", index=False)
    np.savez_compressed(
        destination / "generated_images.npz",
        images=dataset.images,
        labels=dataset.labels,
        positions=dataset.positions,
        thicknesses=dataset.thicknesses,
        occluded=dataset.occluded,
    )
    feature_frame = pd.DataFrame(features, columns=[f"qubit_{index}" for index in range(8)])
    feature_frame.insert(0, "row_id", np.arange(len(feature_frame)))
    feature_frame.to_csv(destination / "patch_features.csv", index=False)
    per_split.to_csv(destination / "metrics_per_split.csv", index=False)
    summary.to_csv(destination / "summary_overall.csv", index=False)
    predictions.to_csv(destination / "test_predictions.csv", index=False)
    splits.to_csv(destination / "split_assignments.csv", index=False)
    subgroup.to_csv(destination / "occlusion_subgroup_metrics.csv", index=False)
    training_history.to_csv(destination / "qcnn_training_history.csv", index=False)
    robustness.to_csv(destination / "qcnn_robustness.csv", index=False)
    wins.to_csv(destination / "split_wins_or_ties.csv", index=False)

    fitted = first_models[QCNN_NAME]
    if not isinstance(fitted, FittedQCNN):
        raise TypeError("first QCNN model has an unexpected type")
    write_json(destination / "qcnn_model.json", fitted.to_dict())
    joblib.dump(
        {name: model for name, model in first_models.items() if name != QCNN_NAME},
        destination / "classical_models.joblib",
        compress=3,
    )
    write_json(
        destination / "saved_models_manifest.json",
        {
            "saved_repeat": 0,
            "qcnn": "qcnn_model.json",
            "classical": "classical_models.joblib",
            "security": "Load the joblib file only from a trusted benchmark run.",
        },
    )
    circuit = build_symbolic_circuit()
    (destination / "qcnn_circuit.txt").write_text(
        str(circuit.draw(output="text", fold=140)) + "\n", encoding="utf-8"
    )
    (destination / "benchmark_report.md").write_text(
        _report(config, summary, wins, robustness, circuit_info), encoding="utf-8"
    )
    save_all_plots(
        destination,
        dataset,
        features,
        summary,
        predictions,
        training_history,
        robustness,
    )
