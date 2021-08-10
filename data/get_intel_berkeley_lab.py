from typing import Any, List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_and_preprocess_data(
    should_smooth: bool = False,
    smoothing_window: int = 100,
    sequence_length: int = 120,
    cut_off_min: int = 5,
    cut_off_max: int = 45,
    should_scale: bool = True,
    data_path: str = "data/datasets/data.txt",
    batch_size: int = 32,
    motes_train: List = [1, 2, 3, 4, 6, 7, 9, 10, 32, 34, 35],
    motes_test: List = [36]
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
    df = pd.read_csv(
        data_path,
        sep=" ",
        lineterminator="\n",
        names=[
            "date",
            "time",
            "epoch",
            "moteid",
            "temperature",
            "humidity",
            "light",
            "voltage",
        ],
    )

    # Clean nans
    df.dropna(inplace=True)

    # Clean outliers
    df.drop(
        df[(df["temperature"] < cut_off_min) | (df["temperature"] > cut_off_max)].index,
        inplace=True,
    )

    # temperature_std = df["temperature"].std()
    lower_bound = df["temperature"].mean() - 3 * df["temperature"].std()
    upper_bound = df["temperature"].mean() + 3 * df["temperature"].std()
    df.drop(
        df[(df["temperature"] < lower_bound) | (df["temperature"] > upper_bound)].index,
        inplace=True,
    )

    def concat_motes(mote_ids: List) -> pd.DataFrame:
        # Concatenate all relevant motes into one dataframe
        tmp_frames = []
        for mote_id in mote_ids:
            tmp_frame = df.loc[df["moteid"] == mote_id]["temperature"]
            tmp_frame = tmp_frame.reset_index(drop=True)
            tmp_frames.append(tmp_frame)
        return pd.concat(tmp_frames, axis=0)

    x_train = concat_motes(motes_train)
    x_test = concat_motes(motes_test)

    if should_smooth:
        x_train = x_train.rolling(window=smoothing_window).mean()[smoothing_window:]
        x_test = x_test.rolling(window=smoothing_window).mean()[smoothing_window:]

    if should_scale:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train.values.reshape(-1, 1))
        x_test = scaler.fit_transform(x_test.values.reshape(-1, 1))

    ###
    # Prepare the data
    ###
    def reshape_inputs(data: Any) -> Any:
        assert sequence_length <= data.shape[0]
        remainder = data.shape[0] % sequence_length
        limit = data.shape[0] - remainder
        data = data[:limit, :]
        n_samples = int(data.shape[0] / sequence_length)

        n_dims = data.shape[1]
        reshaped_data = data.reshape(n_samples, sequence_length, n_dims)

        length = reshaped_data.shape[0]
        cutoff = length % batch_size
        new_length = length - cutoff
        return reshaped_data[:new_length]

    x_train = reshape_inputs(x_train)
    x_test = reshape_inputs(x_test)

    return x_train, x_test
