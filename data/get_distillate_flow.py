from typing import Any, Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_and_preprocess_data(
    should_smooth: bool = False,
    smoothing_window: int = 100,
    sequence_length: int = 120,
    cut_off_min: int = 5,
    cut_off_max: int = 45,
    should_scale: bool = True,
    data_path: str = "data/datasets/distillate_flow.txt",
    batch_size: int = 32
) -> Tuple[Any, Any, dict]:
    """
    """

    # Load, clean and preprocess data
    df = pd.read_csv(
        data_path,
        sep=" ",
        lineterminator="\n",
        names=["Flow"],
    )

    # Clean nans
    df.dropna(inplace=True)

    # Clean outliers
    df.drop(
        df[(df["Flow"] < cut_off_min) | (df["Flow"] > cut_off_max)].index,
        inplace=True,
    )

    # temperature_std = df["temperature"].std()
    lower_bound = df["Flow"].mean() - 3 * df["Flow"].std()
    upper_bound = df["Flow"].mean() + 3 * df["Flow"].std()
    df.drop(
        df[(df["Flow"] < lower_bound) | (df["Flow"] > upper_bound)].index,
        inplace=True,
    )

    x_train = df
    x_test = df

    n_dims = x_train.shape[1]
    assert n_dims == x_test.shape[1]

    if should_smooth:
        x_train = x_train.rolling(window=smoothing_window).mean()[smoothing_window:]
        x_test = x_test.rolling(window=smoothing_window).mean()[smoothing_window:]

    if should_scale:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train.values.reshape(-1, n_dims))
        x_test = scaler.fit_transform(x_test.values.reshape(-1, n_dims))

    ###
    # Prepare the data
    ###
    def reshape_inputs(data: Any, n_dims: int) -> Any:
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

    x_train = reshape_inputs(x_train, n_dims)
    x_test = reshape_inputs(x_test, n_dims)

    return x_train, x_test
