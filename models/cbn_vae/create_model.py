# ==================================================================================================
# Imports etc.
# ==================================================================================================
#  External libraries
# Set seeds
from numpy.random import seed; seed(1)
import tensorflow as tf; tf.random.set_seed(1)
# TensorFlow and keras
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Lambda,
    Flatten,
    Reshape,
    Input,
    Conv2D,
    UpSampling2D
)
# Custom layers
from models.cbn_vae.custom_layers.custom_conv2d_transpose import CustomConv2DTranspose
# from models.cbn_vae.custom_layers.unpooling_with_argmax import UnMaxPoolWithArgmax
from models.cbn_vae.custom_layers.max_pooling_with_argmax import MaxPoolWithArgMax
from models.cbn_vae.custom_layers.sampling import sample_from_latent_space
# Other external libs
import matplotlib.pyplot as plt
# Own code
# from benchmarking.compression.benchmarks.reconstruction_error import get_prms_diff
from utils.plotter import Plotter
plotter = Plotter("cbn_vae", plt, backend="WebAgg")


def create_model(config: dict, batch_size: int = 32, sequence_length: int = 120, n_dims: int = 1) -> Model:
    input_shape = (sequence_length, 1, n_dims)
    encoder_inputs: Input = Input(shape=(sequence_length, 1, n_dims), batch_size=batch_size, name='encoder_input')

    channels, kernel = 24, 12
    out_conv1 = Conv2D(  # None, 10, 1, 24 | None, 10, 1, 24
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(encoder_inputs)
    out_reshape1 = Reshape((-1, 1, 1))(out_conv1)  # None, 240, 1, 1 | None, 60, 1, 1
    out_pool1, mask1 = MaxPoolWithArgMax()(out_reshape1)  # None, 120, 1, 1 | None, 30, 1, 1

    channels, kernel = 12, 9
    out_conv2 = Conv2D(  # | None, 4, 1, 12
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(out_pool1)
    out_reshape2 = Reshape((-1, 1, 1))(out_conv2)  # None, 240, 1, 1 | None, 12, 1 ,4
    out_pool2, mask2 = MaxPoolWithArgMax()(out_reshape2)  # None, 32, 84, 1, 1 | None, 6, 1, 4

    out_max_pool_1, mask3 = MaxPoolWithArgMax()(out_pool2)

    channels, kernel = 12, 5
    out_conv3 = Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(out_max_pool_1)
    out_reshape3 = Reshape((-1, 1, 1))(out_conv3)
    out_pool3, mask4 = MaxPoolWithArgMax()(out_reshape3)

    channels, kernel = 12, 3
    out_conv4 = Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(out_pool3)
    out_reshape4 = Reshape((-1, 1, 1))(out_conv4)
    out_pool4, mask5 = MaxPoolWithArgMax()(out_reshape4)

    out_max_pool_2, mask6 = MaxPoolWithArgMax()(out_pool4)

    out_flatten = Flatten()(out_max_pool_2)
    out_dense1 = Dense(config["dense_nodes"], activation=config["bottleneck_activation"])(out_flatten)

    z_mean = Dense(5, activation=config["bottleneck_activation"], name="z_mean")(out_dense1)
    z_log_var = Dense(5, activation=config["bottleneck_activation"], name="z_log_var")(out_dense1)
    z = Lambda(sample_from_latent_space, output_shape=(5,), name="z")([z_mean, z_log_var])

    encoder_output = z
    # ms = [tf.cast(mask, tf.uint8) for mask in [mask1, mask2, mask3, mask4, mask5, mask6]]
    # ms = [tf.cast(mask, tf.uint8) for mask in [mask6, mask3]]
    ms = []
    encoder = Model(encoder_inputs, [encoder_output, ms])

    # Decoder
    decoder_inputs: Input = Input(shape=z.shape[1:], batch_size=batch_size, name='decoder_input')
    masks = [Input(shape=mask.shape[1:], batch_size=batch_size, dtype=tf.int32, name=f'mask{i+1}')
             for i, mask in enumerate(ms)]

    de_out_dense2 = Dense(
        config["dense_nodes"],
        activation=config["bottleneck_activation"],
        name='dense2')(decoder_inputs)
    de_out_dense3 = Dense(54, activation=config["bottleneck_activation"], name='dense3')(de_out_dense2)
    de_inverse_flatten = Reshape((-1, 1, 1), name='reshape4')(de_out_dense3)

    # de_out_pool5 = UnMaxPoolWithArgmax()(de_inverse_flatten, masks[0])  # 6
    # de_out_pool5 = UnMaxPoolWithArgmax()(de_inverse_flatten, masks[-1])  # 6
    de_out_pool5 = UpSampling2D(size=(2, 1))(de_inverse_flatten)  # 6

    channels, kernel = 12, 3
    # de_out_pool6 = UnMaxPoolWithArgmax()(de_out_pool5, masks[-2])  # 5
    de_out_pool6 = UpSampling2D(size=(2, 1))(de_out_pool5)  # 5
    de_out_reshape5 = Reshape(out_conv4.shape[1:])(de_out_pool6)
    de_out_transcov1 = CustomConv2DTranspose(
        channels, kernel, out_pool3.shape, activation=config["decoder_activation"], name='out_transcov1'
    )(de_out_reshape5)

    channels, kernel = 12, 5
    # de_out_pool7 = UnMaxPoolWithArgmax()(de_out_transcov1, masks[-3])  # 4
    de_out_pool7 = UpSampling2D(size=(2, 1))(de_out_transcov1)  # 4
    de_out_reshape6 = Reshape(out_conv3.shape[1:])(de_out_pool7)
    de_out_transcov2 = CustomConv2DTranspose(
        channels,
        kernel,
        out_max_pool_1.shape,
        activation=config["decoder_activation"],
        name='out_transcov2'
    )(de_out_reshape6)

    # de_out_pool8 = UnMaxPoolWithArgmax()(de_out_transcov2, masks[-4])  # 3
    # de_out_pool8 = UnMaxPoolWithArgmax()(de_out_transcov2, masks[1])  # 3
    de_out_pool8 = UpSampling2D(size=(2, 1))(de_out_transcov2)

    channels, kernel = 12, 9
    # de_out_pool9 = UnMaxPoolWithArgmax()(de_out_pool8, masks[-5])  # 2
    de_out_pool9 = UpSampling2D(size=(2, 1))(de_out_pool8)  # 2
    de_out_reshape7 = Reshape(out_conv2.shape[1:])(de_out_pool9)
    de_out_transcov3 = CustomConv2DTranspose(
        channels, kernel, out_pool1.shape, activation=config["decoder_activation"], name='out_transcov3'
    )(de_out_reshape7)

    channels, kernel = 24, 12
    # de_out_pool10 = UnMaxPoolWithArgmax()(de_out_transcov3, masks[-6])  # 1
    de_out_pool10 = UpSampling2D(size=(2, 1))(de_out_transcov3)  # 1
    de_out_reshape8 = Reshape(out_conv1.shape[1:])(de_out_pool10)
    de_out_transcov4 = CustomConv2DTranspose(
        channels, kernel, encoder_inputs.shape, activation=config["decoder_activation"],
        n_filter_dims=n_dims, name='out_transcov4')(de_out_reshape8)
    decoder_outputs = de_out_transcov4
    decoder = Model([decoder_inputs, masks], decoder_outputs)

    # Autoencoder
    model_input = Input(shape=input_shape, name='autoencoder_input')
    encoder_outputs = encoder(model_input)
    decoded = decoder(encoder_outputs)
    model = Model(model_input, decoded, name='cbn_vae')

    adam_optimizer = keras.optimizers.Adam(lr=config["lr"])
    sgd_optimizer = keras.optimizers.SGD(lr=config["lr"], momentum=config["sgd_momentum"])
    optimizer = adam_optimizer if config['optimizer'] == 'Adam' else sgd_optimizer

    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=[],
    )
    return encoder, decoder, model
