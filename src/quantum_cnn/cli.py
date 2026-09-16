"""Command-line benchmark and trusted saved-model inference.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .benchmark import run_benchmark
from .config import BenchmarkConfig
from .data import make_single_image
from .inference import load_image, predict_saved_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum-cnn",
        description="Train and compare an eight-qubit QCNN image classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="run the complete benchmark")
    benchmark.add_argument("--config", help="JSON configuration; omit for defaults")
    benchmark.add_argument("--samples", type=int, help="generated image count")
    benchmark.add_argument("--repeats", type=int, help="stratified split count")
    benchmark.add_argument("--qcnn-epochs", type=int, help="QCNN training epochs")
    benchmark.add_argument("--seed", type=int, help="experiment random seed")
    benchmark.add_argument("--output-root", help="artifact root")

    predict = subparsers.add_parser("predict", help="score one image with saved models")
    predict.add_argument("--artifacts", required=True, help="trusted completed run directory")
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="8×8 .npy or comma-separated .csv image")
    source.add_argument("--orientation", choices=["vertical", "horizontal"])
    predict.add_argument("--position", type=int, default=3)
    predict.add_argument("--thickness", type=int, choices=[1, 2], default=1)
    predict.add_argument("--noise", type=float, default=0.12)
    predict.add_argument("--seed", type=int, default=7)
    predict.add_argument("--occluded", action="store_true")
    predict.add_argument("--output", help="optional prediction CSV")
    predict.add_argument("--save-image", help="optional generated image .npy path")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "benchmark":
        base = (
            BenchmarkConfig.from_json(arguments.config)
            if arguments.config
            else BenchmarkConfig().validate()
        )
        config = base.with_overrides(
            samples=arguments.samples,
            repeats=arguments.repeats,
            qcnn_epochs=arguments.qcnn_epochs,
            random_seed=arguments.seed,
            output_root=arguments.output_root,
        )
        result = run_benchmark(config, print)
        columns = ["model", "accuracy_mean", "accuracy_std", "f1_mean", "roc_auc_mean"]
        print("\nOverall repeated-holdout result")
        print(result.summary[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nArtifacts: {result.output_directory}")
        return 0

    image = (
        load_image(arguments.image)
        if arguments.image
        else make_single_image(
            arguments.orientation,
            arguments.position,
            arguments.thickness,
            arguments.noise,
            arguments.seed,
            arguments.occluded,
        )
    )
    predictions = predict_saved_models(arguments.artifacts, image)
    print(predictions.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output, index=False)
        print(f"Predictions: {output}")
    if arguments.save_image:
        destination = Path(arguments.save_image)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, image, allow_pickle=False)
        print(f"Image: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
