"""Validated experiment configuration.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """All settings required to reproduce one benchmark run."""

    samples: int = 240
    image_size: int = 8
    image_noise: float = 0.16
    occlusion_probability: float = 0.25
    repeats: int = 3
    test_size: float = 0.30
    qcnn_epochs: int = 80
    qcnn_learning_rate: float = 0.05
    qcnn_l2: float = 0.0005
    mlp_hidden_units: int = 8
    mlp_max_iterations: int = 2_000
    shot_count: int = 256
    shot_trials: int = 10
    parameter_noise: float = 0.04
    noise_trials: int = 10
    random_seed: int = 42
    split_seed_offset: int = 10_000
    output_root: str = "artifacts"

    def validate(self) -> BenchmarkConfig:
        if not 80 <= self.samples <= 2_000:
            raise ValueError("samples must be between 80 and 2,000")
        if self.image_size != 8:
            raise ValueError("image_size must be 8 for the fixed eight-patch encoder")
        if not 0 <= self.image_noise <= 0.5:
            raise ValueError("image_noise must be in [0, 0.5]")
        if not 0 <= self.occlusion_probability <= 1:
            raise ValueError("occlusion_probability must be in [0, 1]")
        if not 1 <= self.repeats <= 20:
            raise ValueError("repeats must be between 1 and 20")
        if not 0.15 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.15 and 0.5")
        if not 5 <= self.qcnn_epochs <= 500:
            raise ValueError("qcnn_epochs must be between 5 and 500")
        if not 0 < self.qcnn_learning_rate <= 0.25:
            raise ValueError("qcnn_learning_rate must be in (0, 0.25]")
        if not 0 <= self.qcnn_l2 <= 0.1:
            raise ValueError("qcnn_l2 must be in [0, 0.1]")
        if not 2 <= self.mlp_hidden_units <= 128:
            raise ValueError("mlp_hidden_units must be between 2 and 128")
        if not 100 <= self.mlp_max_iterations <= 20_000:
            raise ValueError("mlp_max_iterations must be between 100 and 20,000")
        if not 16 <= self.shot_count <= 100_000:
            raise ValueError("shot_count must be between 16 and 100,000")
        if not 1 <= self.shot_trials <= 100:
            raise ValueError("shot_trials must be between 1 and 100")
        if not 0 <= self.parameter_noise <= 1:
            raise ValueError("parameter_noise must be in [0, 1]")
        if not 1 <= self.noise_trials <= 100:
            raise ValueError("noise_trials must be between 1 and 100")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.split_seed_offset < 1_000:
            raise ValueError("split_seed_offset must be at least 1,000")
        if not self.output_root.strip():
            raise ValueError("output_root cannot be empty")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> BenchmarkConfig:
        values = {name: value for name, value in overrides.items() if value is not None}
        return replace(self, **values).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> BenchmarkConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        return cls(**values).validate()
