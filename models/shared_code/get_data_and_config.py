from typing import Any, Tuple
from data.get_intel_berkeley_lab import read_and_preprocess_data as read_intel
from data.get_distillate_flow import read_and_preprocess_data as read_distillate
from utils.terminal_colorizer import printc


def get_data_and_config(
        dataset_name: str, sequence_length: int, batch_size: int,
        train_test_cutoff: int,
        is_multivariate: bool = False,
) -> Tuple[
        Any, Any, int, int
]:
    if dataset_name == 'intel':
        x_train, x_test = read_intel(
            sequence_length=sequence_length,
            batch_size=batch_size,
            motes_train=[7],
            motes_test=[7],
            selected_dimensions=['temperature', 'humidity', 'light', 'voltage'] if is_multivariate else ['temperature']
        )

    elif dataset_name == 'distillate_flow':
        if is_multivariate:
            raise Exception(f'Dataset "{dataset_name}" does not support multivariate data')
        x_train, x_test = read_distillate(
            sequence_length=sequence_length,
            batch_size=batch_size
        )

    x_train = x_train[:train_test_cutoff, :, :]
    x_test = x_test[train_test_cutoff:, :, :]

    printc(f"[train] NUM EXAMPLES | SEQUENCE LENGTH | NUM DIMENSIONS: {x_train.shape[0]} \
    | {x_train.shape[1]} | {x_train.shape[2]}", color="blue")
    printc(f"[test]  NUM EXAMPLES | SEQUENCE LENGTH | NUM DIMENSIONS: {x_test.shape[0]} \
    | {x_test.shape[1]} | {x_test.shape[2]}", color="blue")
    return x_train, x_test, sequence_length, batch_size
