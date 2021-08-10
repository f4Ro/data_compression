from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# If you change anything here please make sure that you're using the
# "tensorflow.keras" imports instead of the "keras" imports.
# Somehow keras tensors are treated differently than tf tensors

original_shape = (10,)
encoded_shape = (1,)

# Encoder
enc_inp = Input(shape=original_shape, name='encoder_input')
enc_out = Dense(units=1, activation='relu')(enc_inp)
encoder = Model(enc_inp, enc_out)

# Decoder
dec_inp = Input(shape=encoded_shape, name='decoder_input')
dec_out = Dense(units=1, activation='relu')(dec_inp)
decoder = Model(dec_inp, dec_out)

# Autoencoder
ae_inp = Input(shape=original_shape)
encoded = encoder(ae_inp)
decoded = decoder(encoded)
model = Model(ae_inp, decoded)
model.compile(loss='mse', optimizer='Adam')
