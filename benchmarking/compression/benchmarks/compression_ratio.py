import tensorflow as tf

from typing import Any, List
from functools import reduce


def reduce_multiply(array: List) -> int:
    return reduce(lambda x, y: x * y, array)


def get_compression_ratio(original: Any, encoder_output: Any) -> float:
    """
    Get the compression ratio of the model based on the inputs and its outputs on the bottleneck.
    The CR describes the factor by which the data is compressed in size.

    For this function it is assumed that all numbers in the data share their data type (typically
    that is float32, which is 32bit/4bytes of data). Thus, for the ratio of size, only the volumes
    of the two tensors have to be calculated and divided.
    """

    # Ignore the first dimension (batch size) as it is always the same in both volumes and therefore
    # cancels out. In addition, this will cause an error if it is None
    original_volume = reduce_multiply(original.shape[1:])
    encoded_volume = reduce_multiply(encoder_output.shape[1:])
    if (encoded_volume > original_volume):
        raise Exception(
            f'Encoded data is bigger than original ({encoded_volume} > {original_volume}). \
            Have you mixed up the order of arguments?'
        )

    return original_volume / encoded_volume


if __name__ == '__main__':
    orig = tf.ones((1, 10))
    comp = tf.ones((1, 1))
    print(get_compression_ratio(orig, comp))
