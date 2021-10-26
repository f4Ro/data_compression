import json
from operator import itemgetter
import tensorflow as tf

import matplotlib.pyplot as plt
from models.shared_code.get_data_and_config import get_data_and_config
from utils.plotter import Plotter

from benchmarking.benchmarks import run_benchmarks_once
from benchmarking.model_configurations import configurations


def conduct_full_experiment(dataset: str, model: str, is_multivariate: bool, num_iterations: int = 10):
    assert dataset in ['intel', 'distillate_flow']
    assert model in ['cbn_vae', 'rnn']
    plotter = Plotter('experiments', plt)

    avg_results = conduct_single_experiment(plotter, dataset, model, is_multivariate, iteration_number=1)
    for i in range(0, num_iterations - 1):
        result = conduct_single_experiment(plotter, dataset, model, is_multivariate, iteration_number=i + 2)
        avg_results = {k: (result[k] + avg_results[k]) / 2 for k in set(result)}

    with open(f'results/experiments/{model}/avg_results.json', 'w') as file:
        file.write(json.dumps(avg_results))


def conduct_single_experiment(
        plotter: Plotter, dataset_name: str, model_name: str, is_multivariate: bool, iteration_number: int = 1):
    config = configurations[model_name]
    create_model, batch_size, sequence_length = itemgetter('create_model', 'batch_size', 'sequence_length')(config)
    num_epochs, default_config = itemgetter('num_epochs', 'default_config')(config)
    train_test_cutoff = config['train_test_cutoff'][dataset_name]

    x_train, x_test, _, _ = get_data_and_config(
        dataset_name, sequence_length, batch_size, train_test_cutoff, is_multivariate)
    n_dims = x_train.shape[2]
    x_train = tf.reshape(x_train, (-1, sequence_length, 1, n_dims))
    x_test = tf.reshape(x_test, (-1, sequence_length, 1, n_dims))
    encoder, decoder, model = create_model(default_config, n_dims=n_dims)
    model.fit(
        x_train, x_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_test, x_test),
        validation_batch_size=batch_size,
        callbacks=[],
        verbose=0
    )
    preds = model.predict(x_test, batch_size=batch_size)
    plt.plot(preds.reshape(-1), label='Reconstruction')
    plt.plot(x_test.numpy().reshape(-1), label='Original')
    plt.legend()
    plotter(f'{model_name}-{dataset_name}-{iteration_number}', sub_path=f'/{model_name}/{iteration_number}')

    results = run_benchmarks_once(encoder, decoder, model, x_test, batch_size, verbose=False)
    with open(f'results/experiments/{model_name}/{iteration_number}/{model_name}-{dataset_name}-{iteration_number}.json', 'w') as outfile:
        outfile.write(json.dumps(results))
    return results
