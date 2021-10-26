import pywt
import numpy as np
import tensorflow as tf

# from benchmarking.compression.benchmarks.compression_ratio import get_compression_ratio
# from benchmarking.compression.benchmarks.reconstruction_error import get_reconstruction_error
from typing import Any, List, Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.get_intel_berkeley_lab import read_and_preprocess_data

x_train, x_test = read_and_preprocess_data()

wavelet = "db1"

def get_prms_diff(original: Any, prediction: Any, to_numpy: bool = True) -> Any:
    diff = tf.reduce_sum(tf.square(tf.subtract(original, prediction)))
    sq = tf.reduce_sum(tf.square(original))
    prms = 100 * (tf.sqrt(tf.divide(diff, sq + 0.0e-6)))

    return prms.numpy() if to_numpy else prms

def dwt(time_series, keep):
    reconstruction = None
    compression_ratio = (1 / keep)

    for time_step in time_series:
        time_step = np.squeeze(time_step)
        coeffs = pywt.wavedec2(time_step, wavelet=wavelet, level=4)

        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

        Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

        number = int(np.floor((1 - keep) * len(Csort)))
        thresh = Csort[number]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr * ind

        coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format="wavedec2")
        Arecon = pywt.waverec2(coeffs_filt, wavelet=wavelet)
        if reconstruction is None:
            reconstruction = Arecon
        else:
            reconstruction = np.concatenate((reconstruction, Arecon), axis=None)

    diff = get_prms_diff(time_series.reshape(-1), reconstruction)
    print(diff, "% for compression ratio of", compression_ratio)

    return reconstruction

dwt(x_train, 0.1)
dwt(x_train, 0.0667)
dwt(x_train, 0.05)
