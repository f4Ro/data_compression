from typing import Any, Tuple
from data.get_intel_berkeley_lab import read_and_preprocess_data
from utils.terminal_colorizer import printc


def get_data_and_config(
        dataset_name: str, sequence_length: int, batch_size: int, is_multivariate: bool = False) -> Tuple[
        Any, Any, int, int]:
    if dataset_name == 'intel':
        x_train, x_test = read_and_preprocess_data(
            sequence_length=sequence_length,
            batch_size=batch_size,
            motes_train=[7],
            motes_test=[7],
            selected_dimensions=['temperature', 'humidity', 'light', 'voltage'] if is_multivariate else ['temperature']
        )
        x_train = x_train[:1900, :, :]
        x_test = x_test[1900:, :, :]

    printc(f"[train] NUM EXAMPLES | SEQUENCE LENGTH | NUM DIMENSIONS: {x_train.shape[0]} \
    | {x_train.shape[1]} | {x_train.shape[2]}", color="blue")
    printc(f"[test]  NUM EXAMPLES | SEQUENCE LENGTH | NUM DIMENSIONS: {x_test.shape[0]} \
    | {x_test.shape[1]} | {x_test.shape[2]}", color="blue")
    return x_train, x_test, sequence_length, batch_size
