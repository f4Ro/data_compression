import tensorflow as tf
from typing import Any


def get_reconstruction_error(original: Any, reconstruction: Any) -> float:
    """
    Get the percentual root mean square difference between two tensors.
    For this, first the RMS-Error between the tensors is calculated and then
    'normalized' to the percentual value (how much of the mean it deviates).
    This is asymmetric, meaning that the result, you'll get differs from the order
    in which you provide the arguments.
    Always put the "ground truth" first.
    """
    return get_prms_diff(original, reconstruction)


def get_prms_diff(original: Any, prediction: Any, to_numpy: bool = True) -> Any:
    diff = tf.reduce_sum(tf.square(tf.subtract(original.reshape(-1), prediction.reshape(-1))))
    sq = tf.reduce_sum(tf.square(original))
    prms = 100 * (tf.sqrt(tf.divide(diff, sq + 0.e-6)))

    return prms.numpy() if to_numpy else prms


if __name__ == '__main__':
    t1 = tf.ones((1, 10))
    t2 = tf.ones((1, 10))

    t3 = tf.zeros((1, 10))
    print(get_reconstruction_error(t1, t2))  # Should be 0.0
    print(get_reconstruction_error(t1, t3))  # Should be 100.0
