import pywt
import numpy as np
import tensorflow as tf

# from benchmarking.compression.benchmarks.compression_ratio import get_compression_ratio
# from benchmarking.compression.benchmarks.reconstruction_error import get_reconstruction_error
from typing import Any
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


wavelet = "db1"

batch_size: int = 1
sequence_length: int = 120
# motes_train = [7]


def read_and_preprocess_data(
    should_smooth: bool = False,
    smoothing_window: int = 100,
    sequence_length: int = 120,
    # cut_off_min: int = 5,
    # cut_off_max: int = 45,
    should_scale: bool = True,
    data_path: str = "/work/data/measurement_86.txt",
    batch_size: int = 32,
    # motes_train: List = [1, 2, 3, 4, 6, 7, 9, 10, 32, 34, 35],
    # motes_test: List = [36],
) -> Any:
    """
    Load the temperature sensor data of the "Intel Berkeley Research Lab" dataset, clean it and scale it down.

    :parameters:
    cut_off_min(number)   -- threshhold to discard all temperatures below that point
    cut_off_max(number)   -- threshhold to discard all temperatures above that point
    should_scale(boolean) -- switch between min-max-scaling data or not
    data_path(string)     -- path to the file containing all data

    :returns:
    x_train -- numpy array of shape dictated by config and train_range
    x_test  -- numpy array of shape dictated by config and test_range
    config  -- chosen config in case it needs to be reused later on
    """
    # Load, clean and preprocess data
    x_train = pd.read_csv(
        data_path,
        sep=",",
        lineterminator="\n",
        names=[
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
        ],
    )

    # Clean nans
    x_train.dropna(inplace=True)

    # Clean outliers
    # df.drop(
    #     df[(df["temperature"] < cut_off_min) | (df["temperature"] > cut_off_max)].index,
    #     inplace=True,
    # )

    # temperature_std = df["temperature"].std()
    # lower_bound = df["temperature"].mean() - 3 * df["temperature"].std()
    # upper_bound = df["temperature"].mean() + 3 * df["temperature"].std()
    # df.drop(
    #     df[(df["temperature"] < lower_bound) | (df["temperature"] > upper_bound)].index,
    #     inplace=True,
    # )

    # def concat_motes(mote_ids: List) -> pd.DataFrame:
    #     # Concatenate all relevant motes into one dataframe
    #     tmp_frames = []
    #     for mote_id in mote_ids:
    #         tmp_frame = df.loc[df["moteid"] == mote_id][["temperature", "humidity", "light", "voltage"]]
    #         tmp_frame = tmp_frame.reset_index(drop=True)
    #         tmp_frames.append(tmp_frame)
    #     return pd.concat(tmp_frames, axis=0)

    # x_train = concat_motes(motes_train)
    # x_test = concat_motes(motes_test)

    n_dims = x_train.shape[1]
    # assert n_dims == x_test.shape[1]

    if should_smooth:
        x_train = x_train.rolling(window=smoothing_window).mean()[smoothing_window:]
        # x_test = x_test.rolling(window=smoothing_window).mean()[smoothing_window:]

    if should_scale:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train.values.reshape(-1, n_dims))
        # x_test = scaler.fit_transform(x_test.values.reshape(-1, n_dims))

    print(x_train)
    exit()

    ###
    # Prepare the data
    ###
    def reshape_inputs(data: Any, n_dims: int) -> Any:
        assert sequence_length <= data.shape[0]
        remainder = data.shape[0] % sequence_length
        limit = data.shape[0] - remainder
        data = data[:limit, :]
        n_samples = int(data.shape[0] / sequence_length)

        reshaped_data = data.reshape(n_samples, sequence_length, n_dims)

        length = reshaped_data.shape[0]
        cutoff = length % batch_size
        new_length = length - cutoff
        return reshaped_data[:new_length]

    x_train = reshape_inputs(x_train, n_dims)
    # x_test = reshape_inputs(x_test, n_dims)
    return x_train


x_train = read_and_preprocess_data(
    sequence_length=sequence_length,
    batch_size=batch_size
)


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
