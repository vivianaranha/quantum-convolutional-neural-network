"""Deterministic synthetic image generation and spatial compression.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ImageDataset:
    images: np.ndarray
    labels: np.ndarray
    positions: np.ndarray
    thicknesses: np.ndarray
    occluded: np.ndarray

    def validate(self) -> ImageDataset:
        if self.images.ndim != 3 or self.images.shape[1:] != (8, 8):
            raise ValueError("images must have shape (samples, 8, 8)")
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")
        if (
            not np.isfinite(self.images).all()
            or not ((self.images >= 0) & (self.images <= 1)).all()
        ):
            raise ValueError("image pixels must be finite values in [0, 1]")
        if set(np.unique(self.labels)) - {0, 1}:
            raise ValueError("labels must contain only 0 and 1")
        return self

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | bool | str]] = []
        for row_id, (image, label, position, thickness, occluded) in enumerate(
            zip(
                self.images,
                self.labels,
                self.positions,
                self.thicknesses,
                self.occluded,
                strict=True,
            )
        ):
            row: dict[str, float | int | bool | str] = {
                "row_id": row_id,
                "label": int(label),
                "class_name": "horizontal" if label else "vertical",
                "position": int(position),
                "thickness": int(thickness),
                "occluded": bool(occluded),
            }
            row.update({f"pixel_{r}_{c}": float(image[r, c]) for r in range(8) for c in range(8)})
            rows.append(row)
        return pd.DataFrame(rows)


def make_single_image(
    orientation: str,
    position: int,
    thickness: int,
    noise: float,
    random_seed: int,
    occluded: bool = False,
) -> np.ndarray:
    """Create one 8×8 vertical- or horizontal-line image."""

    normalized = orientation.strip().lower()
    if normalized not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if not 0 <= position <= 7:
        raise ValueError("position must be between 0 and 7")
    if thickness not in {1, 2}:
        raise ValueError("thickness must be 1 or 2")
    if not 0 <= noise <= 0.5:
        raise ValueError("noise must be in [0, 0.5]")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")

    rng = np.random.default_rng(random_seed)
    image = rng.normal(0.06, noise * 0.35, size=(8, 8))
    start = min(position, 8 - thickness)
    intensity = rng.uniform(0.78, 1.0)
    if normalized == "vertical":
        image[:, start : start + thickness] += intensity
    else:
        image[start : start + thickness, :] += intensity

    # A faint nuisance patch prevents the task from being a single-pixel shortcut.
    patch_row = int(rng.integers(0, 7))
    patch_col = int(rng.integers(0, 7))
    image[patch_row : patch_row + 2, patch_col : patch_col + 2] += rng.uniform(0.0, 0.22)
    if occluded:
        block_row = int(rng.integers(0, 7))
        block_col = int(rng.integers(0, 7))
        image[block_row : block_row + 2, block_col : block_col + 2] *= rng.uniform(0.0, 0.25)
    image += rng.normal(0.0, noise, size=(8, 8))
    return np.clip(image, 0.0, 1.0)


def make_image_dataset(
    samples: int,
    noise: float,
    occlusion_probability: float,
    random_seed: int,
) -> ImageDataset:
    """Generate a balanced and shuffled line-orientation dataset."""

    if samples < 4:
        raise ValueError("samples must be at least 4")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    rng = np.random.default_rng(random_seed)
    labels = np.arange(samples, dtype=int) % 2
    rng.shuffle(labels)
    positions = rng.integers(0, 8, size=samples)
    thicknesses = rng.integers(1, 3, size=samples)
    occluded = rng.random(samples) < occlusion_probability
    image_seeds = rng.integers(0, np.iinfo(np.int32).max, size=samples)
    images = np.stack(
        [
            make_single_image(
                "horizontal" if label else "vertical",
                int(position),
                int(thickness),
                noise,
                int(image_seed),
                bool(is_occluded),
            )
            for label, position, thickness, is_occluded, image_seed in zip(
                labels, positions, thicknesses, occluded, image_seeds, strict=True
            )
        ]
    )
    return ImageDataset(images, labels, positions, thicknesses, occluded).validate()


def extract_patch_features(images: np.ndarray) -> np.ndarray:
    """Compress 8×8 images into a spatial 4×2 grid of eight patch means."""

    values = np.asarray(images, dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (8, 8):
        raise ValueError("images must have shape (samples, 8, 8)")
    if not np.isfinite(values).all():
        raise ValueError("images must contain only finite values")
    clipped = np.clip(values, 0.0, 1.0)
    return clipped.reshape(len(clipped), 4, 2, 2, 4).mean(axis=(2, 4)).reshape(len(clipped), 8)
