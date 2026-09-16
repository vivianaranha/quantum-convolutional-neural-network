"""Trusted saved-model image inference.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import extract_patch_features
from .models import FittedQCNN


def validate_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    if values.shape != (8, 8):
        raise ValueError("image must have shape (8, 8)")
    if not np.isfinite(values).all():
        raise ValueError("image must contain only finite values")
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError("image pixels must be in [0, 1]")
    return values


def load_image(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        values = np.load(source, allow_pickle=False)
    elif source.suffix.lower() == ".csv":
        values = np.loadtxt(source, delimiter=",")
    else:
        raise ValueError("image file must use .npy or .csv")
    return validate_image(values)


def predict_saved_models(directory: str | Path, image: np.ndarray) -> pd.DataFrame:
    """Score one 8×8 image with models from a trusted completed run."""

    source = Path(directory)
    values = validate_image(image)
    qcnn_values = json.loads((source / "qcnn_model.json").read_text(encoding="utf-8"))
    classical_models = joblib.load(source / "classical_models.joblib")
    features = extract_patch_features(values)
    models = {
        "Quantum Convolutional Neural Network": FittedQCNN.from_dict(qcnn_values).model,
        **classical_models,
    }
    rows = []
    for name, model in models.items():
        probability = model.predict_proba(features)[0]
        rows.append(
            {
                "model": name,
                "prediction": "horizontal" if probability[1] >= 0.5 else "vertical",
                "probability_vertical": float(probability[0]),
                "probability_horizontal": float(probability[1]),
            }
        )
    return pd.DataFrame(rows)
