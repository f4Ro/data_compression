from typing import Any
from tensorflow.keras.models import Model

from benchmarking.compression.compression_benchmarks import run_compression_benchmarks
from benchmarking.performance.performance_benchmarks import run_performance_benchmarks


def run_benchmarks(
        encoder: Model, decoder: Model, model: Model, data: Any, batch_size: int, verbose: bool = False) -> dict:

    # Always put performance benchmarks last because in the process the model graph is frozen
    # which invalidates the encoder & decoder submodels
    compression_results = run_compression_benchmarks(encoder, model, data, batch_size, verbose=verbose)
    performance_results = run_performance_benchmarks(encoder, decoder, model, data, batch_size, verbose=verbose)

    return {
        **compression_results,
        **performance_results
    }


# if __name__ == '__main__':
#     # from models.dummy_model import encoder, decoder, model
#     # data = tf.ones((1, 10))
#     # print(run_benchmarks(encoder, decoder, model, data))
#     from models.rnn.rnn import create_model
#     encoder, decoder, model = create_model(20, 1, 1)
#     data = tf.ones((1, 20, 1))
#     print(run_benchmarks(encoder, decoder, model, data, batch_size))
