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
    original_reshaped = tf.reshape(tf.cast(original, tf.float64), (-1))
    prediction_reshaped = tf.reshape(tf.cast(prediction, tf.float64), (-1))

    diff = tf.reduce_sum(tf.square(tf.subtract(original_reshaped, prediction_reshaped)))
    sq = tf.reduce_sum(tf.square(original))
    prms = 100 * (tf.sqrt(tf.divide(diff, sq + 0.e-6)))

    return prms.numpy() if to_numpy else prms
