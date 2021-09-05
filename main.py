"""
Run a given pretrained model
"""

from models.rnn.get_data_and_config import get_data_and_config
from models.rnn.create_model import create_model
from models.rnn.train_model import train_model
from benchmarking.benchmarks import run_benchmarks
from utils.plotter import Plotter
import matplotlib.pyplot as plt
plotter = Plotter("RNN", plt, backend="WebAgg")


x_train, x_test, sequence_length, batch_size = get_data_and_config('intel', 20, 1, is_multivariate=True)
encoder, decoder, model = create_model(sequence_length, x_train.shape[2], batch_size)
train_model(model, x_train, x_test, 10, 1, plotter)
print(run_benchmarks(encoder, decoder, model, x_test, batch_size))
