from typing import Any, List
from functools import reduce

dtype_mapping = {
    'uint8': 8,
    'int16': 16,
    'int32': 32,
    'float32': 32,
    'float64': 64,
    "<dtype: 'float64'>": 64
}


def reduce_multiply(array: List, dtype: Any) -> int:
    return reduce(lambda x, y: x * y, array) * dtype_mapping[dtype.__str__()]


def get_encoded_volume(list_or_tensor: Any) -> int:
    if isinstance(list_or_tensor, list):  # if model returns more than one tensor
        return sum(list(map(get_encoded_volume, list_or_tensor)))
    else:
        return reduce_multiply(list_or_tensor.shape[1:], list_or_tensor.dtype)


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
    original_volume = reduce_multiply(original.shape[1:], original.dtype)
    encoded_volume = get_encoded_volume(encoder_output)

    if (encoded_volume > original_volume):
        raise Exception(
            f'Encoded data is bigger than original ({encoded_volume} > {original_volume}). \
            Have you mixed up the order of arguments?'
        )

    return original_volume / encoded_volume
