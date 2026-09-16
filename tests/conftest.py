"""Shared integration fixtures.

Created by School of AI and School of QC.
"""

import pytest

from quantum_cnn import BenchmarkConfig, run_benchmark


@pytest.fixture(scope="session")
def tiny_result(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("qcnn-artifacts")
    config = BenchmarkConfig(
        samples=80,
        repeats=1,
        qcnn_epochs=5,
        mlp_max_iterations=100,
        shot_trials=2,
        noise_trials=2,
        random_seed=999,
        output_root=str(output_root),
    )
    return run_benchmark(config)
