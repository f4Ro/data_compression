from tensorflow.keras.models import Model
from typing import Any

from benchmarking.compression.benchmarks.compression_ratio import get_compression_ratio
from benchmarking.compression.benchmarks.reconstruction_error import get_reconstruction_error


def run_compression_benchmarks(encoder: Model, model: Model, data: Any, verbose: bool = False) -> dict:
    """
    Get the relevant compression metrics for a model.
    Check the respective functions for details about each metric.

    This function also calculates and returns the quality score. This is intended to be a single
    evaluation metric, which is simply the ratio of compression to reconstruction.
    """

    reconstruction = model(data)
    encoding = encoder(data)

    compression_ratio = get_compression_ratio(data, encoding)
    reconstruction_error = get_reconstruction_error(data, reconstruction)
    quality_score = compression_ratio / reconstruction_error

    return {
        'compression_ratio': compression_ratio,
        'reconstruction_error': reconstruction_error,
        'quality_score': quality_score
    }
