from tensorflow.keras import layers
import tensorflow as tf


class MaxPoolWithArgMax(layers.Layer):
    def __init__(self, **kwargs):
        super(MaxPoolWithArgMax, self).__init__(**kwargs)

    def call(self, inputs, stride=2):
        _, index_mask = tf.nn.max_pool_with_argmax(
            inputs, ksize=[1, stride, 1, 1], strides=[1, stride, 1, 1], padding="SAME"
        )
        # Stop the back propagation of the mask gradient calculation
        # bc of argmax mask has shape [old_shape[0],old_shape[1], old_shape[2]/stride, old_shape[3]]
        index_mask = tf.stop_gradient(index_mask)

        # Calculating the maximum pooling operation
        net = tf.nn.max_pool(inputs, ksize=[1, stride, 1, 1], strides=[1, stride, 1, 1], padding="SAME")

        return net, index_mask
