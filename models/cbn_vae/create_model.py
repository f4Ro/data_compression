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
)
# Custom layers
from models.shared_code.get_data_and_config import get_data_and_config
from models.cbn_vae.custom_layers.custom_conv2d_transpose import CustomConv2DTranspose
from models.cbn_vae.custom_layers.unpooling_with_argmax import UnMaxPoolWithArgmax
from models.cbn_vae.custom_layers.max_pooling_with_argmax import MaxPoolWithArgMax
from models.cbn_vae.custom_layers.sampling import sample_from_latent_space
# Other external libs
import matplotlib.pyplot as plt
# Own code
from benchmarking.benchmarks import run_benchmarks
from benchmarking.compression.benchmarks.reconstruction_error import get_prms_diff
from utils.plotter import Plotter
plotter = Plotter("cbn_vae", plt, backend="WebAgg")


def create_model(config: dict, batch_size: int = 32, sequence_length: int = 120, n_dims: int = 1) -> Model:
    input_shape = (sequence_length, 1, n_dims)
    encoder_inputs: Input = Input(shape=(sequence_length, 1, n_dims), batch_size=batch_size, name='encoder_input')

    channels, kernel = 24, 12
    out_conv1 = Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(encoder_inputs)
    out_reshape1 = Reshape((-1, 1, 1))(out_conv1)
    out_pool1, mask1 = MaxPoolWithArgMax()(out_reshape1)

    channels, kernel = 12, 9
    out_conv2 = Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(out_pool1)
    out_reshape2 = Reshape((-1, 1, 1))(out_conv2)
    out_pool2, mask2 = MaxPoolWithArgMax()(out_reshape2)

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
    ms = [tf.cast(mask, tf.uint8) for mask in [mask1, mask2, mask3, mask4, mask5, mask6]]
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

    de_out_pool5 = UnMaxPoolWithArgmax()(de_inverse_flatten, masks[-1])  # 6
    decoder_outputs = de_out_pool5

    channels, kernel = 12, 3
    de_out_pool6 = UnMaxPoolWithArgmax()(de_out_pool5, masks[-2])  # 5
    de_out_reshape5 = Reshape(out_conv4.shape[1:])(de_out_pool6)
    de_out_transcov1 = CustomConv2DTranspose(
        channels, kernel, out_pool3.shape, activation=config["decoder_activation"]
    )(de_out_reshape5)

    channels, kernel = 12, 5
    de_out_pool7 = UnMaxPoolWithArgmax()(de_out_transcov1, masks[-3])  # 4
    de_out_reshape6 = Reshape(out_conv3.shape[1:])(de_out_pool7)
    de_out_transcov2 = CustomConv2DTranspose(
        channels,
        kernel,
        out_max_pool_1.shape,
        activation=config["decoder_activation"],
    )(de_out_reshape6)

    de_out_pool8 = UnMaxPoolWithArgmax()(de_out_transcov2, masks[-4])  # 3

    channels, kernel = 12, 9
    de_out_pool9 = UnMaxPoolWithArgmax()(de_out_pool8, masks[-5])  # 2
    de_out_reshape7 = Reshape(out_conv2.shape[1:])(de_out_pool9)
    de_out_transcov3 = CustomConv2DTranspose(
        channels, kernel, out_pool1.shape, activation=config["decoder_activation"]
    )(de_out_reshape7)

    channels, kernel = 24, 12
    de_out_pool10 = UnMaxPoolWithArgmax()(de_out_transcov3, masks[-6])  # 1
    de_out_reshape8 = Reshape(out_conv1.shape[1:])(de_out_pool10)
    de_out_transcov4 = CustomConv2DTranspose(
        channels, kernel, encoder_inputs.shape, activation=config["decoder_activation"]
    )(de_out_reshape8)
    decoder_outputs = de_out_transcov4
    decoder = Model([decoder_inputs, masks], decoder_outputs)

    # # Autoencoder
    model_input = Input(shape=input_shape, name='autoencoder_input')
    encoder_outputs = encoder(model_input)
    decoded = decoder(encoder_outputs)
    model = Model(model_input, decoded)
    model.compile(optimizer='Adam', loss='mse')

    adam_optimizer = keras.optimizers.Adam(lr=config["lr"])
    sgd_optimizer = keras.optimizers.SGD(lr=config["lr"], momentum=config["sgd_momentum"])
    optimizer = adam_optimizer if config['optimizer'] == 'Adam' else sgd_optimizer

    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=[],
    )
    return encoder, decoder, model


if __name__ == "__main__":
    x_train, x_test, sequence_length, batch_size = get_data_and_config('intel', 120, 32)

    encoder, decoder, model = create_model(
        {
            'encoder_activation': 'tanh',
            'decoder_activation': 'tanh',
            'dense_nodes': 33,
            'bottleneck_activation': 'relu',
            'lr': 0.001521356711612709,
            'optimizer': 'Adam',
            'sgd_momentum': 0.5091287212784572
        }
    )

    # from benchmarking.benchmarks import run_benchmarks
    print(run_benchmarks(encoder, decoder, model, x_test, batch_size))
    # history = model.fit(
    #     x_train,
    #     x_train,
    #     batch_size=32,
    #     epochs=100,
    #     validation_data=(x_test, x_test),
    #     shuffle=False,
    #     callbacks=[],
    # ).history

    # train_preds = model.predict(x_train, batch_size=32)
    # test_preds = model.predict(x_test, batch_size=32)

    # prms_diff = get_prms_diff(x_test.reshape(-1, 120, 1, 1), test_preds)

    # # reconstruction, prms_diff = smooth_output(x_test, test_preds, smoothing_window=5)
    # print(f"The percentual-RMS-difference for the configuration after the re-training is {prms_diff}")
