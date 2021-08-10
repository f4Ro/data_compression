from datetime import datetime
from typing import Any

from tensorflow.keras.models import Model


def _get_time(model: Model, data: Any) -> float:
    """
    Measure the time it takes a model to perform a forward pass.

    To gain more insights, the time is measured seperately for compression and
    decompression. This is done by simply retrieving the current timestamp right
    before and after the model performs its calculations and then taking the
    delta of those.
    """
    tik = datetime.now()
    model(data)
    tok = datetime.now()
    return round((tok - tik).total_seconds() * 1000, 2)


def get_times(encoder: Model, decoder: Model, data: Any) -> tuple:
    decoder_data = encoder(data)

    encoder_time = _get_time(encoder, data)
    decoder_time = _get_time(decoder, decoder_data)

    return encoder_time, decoder_time
