from tensorflow.keras import layers
import tensorflow.keras.backend as K


class CustomConv2DTranspose(layers.Layer):
    def __init__(self, filters, kernel_size, output_shape, activation, n_filter_dims=1, **kwargs):
        super(CustomConv2DTranspose, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.output_shape_ = output_shape
        self.activation = {"tanh": K.tanh, "relu": K.relu, "sigmoid": K.sigmoid}[
            activation
        ]

        # Determines the number of channels of the output of the transconv operation
        self.filter_output_channels = n_filter_dims
        self.trainable = True

    def build(self, _):  # (self, input_shape)
        # shape = (height, width, out_channels, in_channels)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, 1, self.filter_output_channels, self.filters),
            initializer="uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(1,), initializer="uniform", trainable=True
        )

    def call(self, inputs, **kwargs):
        strides = (self.kernel_size, 1)
        convoluted_output = K.conv2d_transpose(
            inputs, self.kernel, self.output_shape_, strides=strides, padding="same"
        )

        outputs = convoluted_output + self.bias
        activations = self.activation(outputs)
        return activations

    def compute_output_shape(self, _):  # (self, input_shape)
        return self.output_shape_

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'output_shape': self.output_shape,
            'activation': self.activation
        })
        return config
