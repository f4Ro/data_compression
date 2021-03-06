import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Any

from benchmarking.performance.benchmarks.flops import get_flops
from benchmarking.performance.benchmarks.params import get_params
from benchmarking.performance.benchmarks.memory import get_memory
from benchmarking.performance.benchmarks.time import get_times
from utils.terminal_colorizer import printc


def run_performance_benchmarks(
        encoder: Model, decoder: Model, model: Model, data: Any, batch_size: int, verbose: bool = False) -> dict:
    """
    Run the benchmarks for a given encoder-decoder model.
    :param encoder: The encoder part of the network, provided as a keras model
    :param decoder: The decoder part of the network, provided as a keras model
    :param model: The complete autoencoder, provided as a keras model
    :param data: The data to run the benchmarks with. Run with a batch size of one to mirror production values

    returns:
        Dictionary containing flops, parameter count, memory in bytes,
        compression-, decompression and full forward propagation time in ms
    """
    # The time tracking has to happen before the get_flops because
    # when get_flops builds a graph it invalidates the encoder & decoder models
    encoder_time, decoder_time = get_times(encoder, decoder, data, batch_size)
    flops = get_flops(model)
    params = get_params(model)
    memory = get_memory(model)

    if verbose:
        printc('==========' * 15, color='yellow')
        printc('FINISHED ALL BENCHMARKS - MODEL RESULTS:', color='yellow')
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
