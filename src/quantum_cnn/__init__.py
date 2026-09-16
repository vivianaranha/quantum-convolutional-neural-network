"""Quantum convolutional neural network image benchmark.

Created by School of AI and School of QC.
"""

from .benchmark import BenchmarkResult, run_benchmark
from .config import BenchmarkConfig

__all__ = ["BenchmarkConfig", "BenchmarkResult", "run_benchmark"]
__version__ = "1.0.0"
