import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Any


from benchmarking.performance.benchmarks.time import get_times
from benchmarking.performance.benchmarks.flops import get_flops
from benchmarking.performance.benchmarks.params import get_params
from benchmarking.performance.benchmarks.memory import get_memory


def run_performance_benchmarks(encoder: Model, decoder: Model, model: Model, data: Any, verbose: bool = False) -> dict:
    """
    Run benchmarks on the performance of the model.
    Check the individual functions for more details on how the metrics are measured.
    """
    # The time tracking has to happen before the get_flops because
    # when get_flops builds a graph it invalidates the encoder&decoder models
    encoder_time, decoder_time = get_times(encoder, decoder, data)
    flops = get_flops(model)
    params = get_params(model)
    memory = get_memory(model)

    if verbose:
        print('==========' * 15)
        print('MODEL RESULTS')
        print('flops:', flops)
        print('params:', params)
        print('memory [bytes]:', memory)
        print('compression time [ms]:')
        print('    encoder:', encoder_time)
        print('    decoder:', decoder_time)
        print('    total:', encoder_time + decoder_time)
    return {
        'flops': flops,
        'params': params,
        'memory': memory,
        'compression_time': encoder_time,
        'decompression_time': decoder_time,
        'total_time': encoder_time + decoder_time
    }


if __name__ == '__main__':
    from models.dummy_model import encoder, decoder, model
    data = tf.ones((1, 10))
    run_performance_benchmarks(encoder, decoder, model, data, verbose=True)
