"""Synthetic image and feature tests.

Created by School of AI and School of QC.
"""

import numpy as np
import pytest

from quantum_cnn.data import extract_patch_features, make_image_dataset, make_single_image


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_single_image_shape_and_range(orientation: str) -> None:
    image = make_single_image(orientation, 3, 1, 0.1, 7)
    assert image.shape == (8, 8)
    assert ((image >= 0) & (image <= 1)).all()


def test_single_image_is_deterministic() -> None:
    first = make_single_image("vertical", 2, 2, 0.1, 9, True)
    second = make_single_image("vertical", 2, 2, 0.1, 9, True)
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"orientation": "diagonal"}, "orientation"),
        ({"position": 8}, "position"),
        ({"thickness": 3}, "thickness"),
        ({"noise": 0.6}, "noise"),
        ({"random_seed": -1}, "non-negative"),
    ],
)
def test_single_image_validation(kwargs: dict[str, object], message: str) -> None:
    values = {
        "orientation": "vertical",
        "position": 3,
        "thickness": 1,
        "noise": 0.1,
        "random_seed": 1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        make_single_image(**values)


def test_dataset_is_balanced_and_deterministic() -> None:
    first = make_image_dataset(100, 0.16, 0.25, 4)
    second = make_image_dataset(100, 0.16, 0.25, 4)
    assert np.bincount(first.labels).tolist() == [50, 50]
    np.testing.assert_allclose(first.images, second.images)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_dataset_frame_has_pixels_and_metadata() -> None:
    frame = make_image_dataset(10, 0.1, 0.2, 5).to_frame()
    assert len(frame) == 10
    assert {"row_id", "label", "class_name", "occluded", "pixel_0_0", "pixel_7_7"} <= set(
        frame.columns
    )


def test_patch_features_have_expected_shape_and_mean() -> None:
    image = np.zeros((8, 8))
    image[:2, :4] = 1.0
    features = extract_patch_features(image)
    assert features.shape == (1, 8)
    np.testing.assert_allclose(features[0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_patch_features_validate_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        extract_patch_features(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="finite"):
        extract_patch_features(np.full((8, 8), np.nan))
