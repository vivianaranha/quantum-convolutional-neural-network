"""Configuration tests.

Created by School of AI and School of QC.
"""

import json

import pytest

from quantum_cnn.config import BenchmarkConfig


def test_default_config_validates() -> None:
    assert BenchmarkConfig().validate().qcnn_epochs == 80


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", 79),
        ("image_size", 4),
        ("image_noise", 0.6),
        ("occlusion_probability", 1.1),
        ("repeats", 0),
        ("test_size", 0.1),
        ("qcnn_epochs", 4),
        ("qcnn_learning_rate", 0.0),
        ("qcnn_l2", -0.1),
        ("mlp_hidden_units", 1),
        ("mlp_max_iterations", 99),
        ("shot_count", 15),
        ("shot_trials", 0),
        ("parameter_noise", -0.1),
        ("noise_trials", 0),
        ("random_seed", -1),
        ("split_seed_offset", 999),
        ("output_root", " "),
    ],
)
def test_invalid_config_values_raise(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(**{field: value}).validate()


def test_json_round_trip(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(BenchmarkConfig().to_dict()), encoding="utf-8")
    assert BenchmarkConfig.from_json(path) == BenchmarkConfig()


def test_json_must_be_object(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        BenchmarkConfig.from_json(path)


def test_overrides_do_not_mutate_original() -> None:
    original = BenchmarkConfig()
    changed = original.with_overrides(samples=100, repeats=1)
    assert changed.samples == 100
    assert changed.repeats == 1
    assert original.samples == 240
