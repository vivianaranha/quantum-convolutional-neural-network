"""Classify one generated image with a trusted completed benchmark run.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import argparse

from quantum_cnn.data import make_single_image
from quantum_cnn.inference import predict_saved_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--orientation", required=True, choices=["vertical", "horizontal"])
    parser.add_argument("--position", type=int, default=3)
    parser.add_argument("--thickness", type=int, choices=[1, 2], default=1)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--occluded", action="store_true")
    arguments = parser.parse_args()
    image = make_single_image(
        arguments.orientation,
        arguments.position,
        arguments.thickness,
        arguments.noise,
        arguments.seed,
        arguments.occluded,
    )
    frame = predict_saved_models(arguments.artifacts, image)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
