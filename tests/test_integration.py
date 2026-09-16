"""End-to-end repository tests.

Created by School of AI and School of QC.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantum_cnn.cli import build_parser, main
from quantum_cnn.data import make_single_image
from quantum_cnn.inference import load_image, predict_saved_models, validate_image


def test_result_dimensions(tiny_result) -> None:
    assert len(tiny_result.per_split) == 4
    assert len(tiny_result.summary) == 4
    assert len(tiny_result.predictions) == 4 * 24
    assert set(tiny_result.per_split["model"]) == {
        "Quantum Convolutional Neural Network",
        "Logistic Regression",
        "RBF SVM",
        "Neural Network",
    }


def test_metrics_are_finite_and_bounded(tiny_result) -> None:
    for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
        assert tiny_result.per_split[metric].between(0, 1).all()
    assert tiny_result.per_split["log_loss"].ge(0).all()
    assert tiny_result.per_split["fit_seconds"].gt(0).all()


def test_complete_artifact_bundle(tiny_result) -> None:
    expected = {
        "accuracy.png",
        "benchmark_report.md",
        "circuit_diagnostics.json",
        "classical_models.joblib",
        "config.json",
        "dependency_versions.json",
        "f1.png",
        "fit_time.png",
        "generated_images.csv",
        "generated_images.npz",
        "image_gallery.png",
        "metrics.json",
        "metrics_per_split.csv",
        "misclassifications.png",
        "occlusion_subgroup_metrics.csv",
        "patch_features.csv",
        "patch_features.png",
        "qcnn_circuit.txt",
        "qcnn_model.json",
        "qcnn_robustness.csv",
        "qcnn_robustness.png",
        "qcnn_training.png",
        "qcnn_training_history.csv",
        "saved_models_manifest.json",
        "split_assignments.csv",
        "summary_overall.csv",
        "test_predictions.csv",
    }
    actual = {path.name for path in tiny_result.output_directory.iterdir()}
    assert expected <= actual


def test_split_artifact_has_no_overlap(tiny_result) -> None:
    frame = pd.read_csv(tiny_result.output_directory / "split_assignments.csv")
    train = set(frame.loc[frame["split"] == "train", "row_id"])
    test = set(frame.loc[frame["split"] == "test", "row_id"])
    assert train.isdisjoint(test)
    assert len(train | test) == 80


def test_saved_model_prediction(tiny_result) -> None:
    image = make_single_image("horizontal", 3, 1, 0.1, 4)
    predictions = predict_saved_models(tiny_result.output_directory, image)
    assert len(predictions) == 4
    assert predictions["probability_horizontal"].between(0, 1).all()
    np.testing.assert_allclose(
        predictions["probability_horizontal"] + predictions["probability_vertical"], 1.0
    )


def test_image_file_loading_and_validation(tmp_path) -> None:
    image = np.full((8, 8), 0.25)
    npy = tmp_path / "image.npy"
    csv = tmp_path / "image.csv"
    np.save(npy, image)
    np.savetxt(csv, image, delimiter=",")
    np.testing.assert_allclose(load_image(npy), image)
    np.testing.assert_allclose(load_image(csv), image)
    with pytest.raises(ValueError, match="shape"):
        validate_image(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="finite"):
        validate_image(np.full((8, 8), np.nan))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_image(np.full((8, 8), 2.0))
    with pytest.raises(ValueError, match="npy or .csv"):
        load_image(tmp_path / "image.png")


def test_prediction_cli_writes_files(tiny_result, tmp_path, capsys) -> None:
    output = tmp_path / "predictions.csv"
    image = tmp_path / "generated.npy"
    status = main(
        [
            "predict",
            "--artifacts",
            str(tiny_result.output_directory),
            "--orientation",
            "vertical",
            "--position",
            "2",
            "--output",
            str(output),
            "--save-image",
            str(image),
        ]
    )
    assert status == 0
    assert len(pd.read_csv(output)) == 4
    assert np.load(image).shape == (8, 8)
    assert "Quantum Convolutional" in capsys.readouterr().out


def test_direct_prediction_script(tiny_result) -> None:
    script = Path(__file__).parents[1] / "scripts" / "predict_saved_models.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifacts",
            str(tiny_result.output_directory),
            "--orientation",
            "horizontal",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Quantum Convolutional" in completed.stdout
    assert "RBF SVM" in completed.stdout


def test_cli_parser_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_metrics_json_contains_attribution(tiny_result) -> None:
    values = json.loads((tiny_result.output_directory / "metrics.json").read_text())
    assert values["attribution"] == "Created by School of AI and School of QC"
    assert len(values["overall_summary"]) == 4


def test_notebook_is_valid_json() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "quickstart.ipynb"
    values = json.loads(path.read_text(encoding="utf-8"))
    assert values["nbformat"] == 4
    assert "Created by School of AI and School of QC" in "".join(values["cells"][0]["source"])


def test_streamlit_app_renders_without_training() -> None:
    path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(path), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Quantum Convolutional Neural Network"
    assert any("Created by School of AI" in caption.value for caption in app.caption)
