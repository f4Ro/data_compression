import tensorflow as tf
from tensorflow.keras.models import Model

from models.cbn_vae.custom_layers.max_pooling_with_argmax import MaxPoolWithArgMax
from models.cbn_vae.custom_layers.unpooling_with_argmax import UnMaxPoolWithArgmax
from models.cbn_vae.custom_layers.custom_conv2d_transpose import CustomConv2DTranspose

custom_objects = {
    'MaxPoolWithArgMax': MaxPoolWithArgMax,
    'UnMaxPoolWithArgmax': UnMaxPoolWithArgmax,
    'CustomConv2DTranspose': CustomConv2DTranspose
}


def get_flops(model: Model, path: str = './build/model.h5') -> int:
    """
    Get the number of floating point operations that a model has to perform when
    propagating inputs through the network.

    :param model: The keras model to perform the flop count on
    :param path: A path to store the model to when converting it to a TensorFlow
                 graph
    """
    # Save the model for later retrieval
    model.save(path)

    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            # For some reason the model does not need to be used again but only
            # loaded. Passing the model as an argument to the function does not
            # work.
            # Another alternative that worked is creating the model inside of
            # this "with" statement
            _ = tf.keras.models.load_model(path, custom_objects=custom_objects)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops
