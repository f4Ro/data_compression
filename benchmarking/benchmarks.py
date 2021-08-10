from typing import Any
import tensorflow as tf
from tensorflow.keras.models import Model

from benchmarking.compression.compression_benchmarks import run_compression_benchmarks
from benchmarking.performance.performance_benchmarks import run_performance_benchmarks


def run_benchmarks(encoder: Model, decoder: Model, model: Model, data: Any, verbose: bool = False) -> dict:

    # Always put performance benchmarks last because in the process the model graph is frozen
    # which invalidates the encoder & decoder submodels
    compression_results = run_compression_benchmarks(encoder, model, data, verbose=verbose)
    performance_results = run_performance_benchmarks(encoder, decoder, model, data, verbose=verbose)

    return {
        **compression_results,
        **performance_results
    }


if __name__ == '__main__':
    from models.dummy_model import encoder, decoder, model
    data = tf.ones((1, 10))
    print(run_benchmarks(encoder, decoder, model, data))
