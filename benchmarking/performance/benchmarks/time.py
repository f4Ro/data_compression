from datetime import datetime
from typing import Any

from tensorflow.keras.models import Model


def _get_time(model: Model, data: Any, batch_size: int) -> float:
    """
    Measure the time it takes a model to perform a forward pass.

    To gain more insights, the time is measured seperately for compression and
    decompression. This is done by simply retrieving the current timestamp right
    before and after the model performs its calculations and then taking the
    delta of those.
    """
    tik = datetime.now()
    model.predict(data, batch_size=batch_size)
    tok = datetime.now()
    return round((tok - tik).total_seconds() * 1000, 2)  # Get the value in milliseconds


def get_times(encoder: Model, decoder: Model, data: Any, batch_size: int) -> tuple:
    decoder_data = encoder.predict(data, batch_size=batch_size)

    encoder_time = _get_time(encoder, data, batch_size)
    decoder_time = _get_time(decoder, decoder_data, batch_size)

    return encoder_time, decoder_time
