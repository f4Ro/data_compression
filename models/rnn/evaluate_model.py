# ==================================================================================================
# Imports etc.
# ==================================================================================================
from typing import Any
import matplotlib.pyplot as plt
from utils.plotter import Plotter
from tensorflow.keras.models import Model


def evaluate_model(model: Model, history: dict, x_train: Any, x_test: Any, batch_size: int, plotter: Plotter):
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.legend()
    plotter("loss", "/training_progress")

    preds = model.predict(x_train, batch_size=batch_size)
    plt.plot(x_train.reshape(-1), label="original")
    plt.plot(preds.reshape(-1), label="reconstruction")
    plt.legend()
    plotter("train_full", "/evaluation")

    preds_test = model.predict(x_test, batch_size=batch_size)
    plt.plot(x_test.reshape(-1), label="original")
    plt.plot(preds_test.reshape(-1), label="reconstruction")
    plt.legend()
    plotter("test_full", "/evaluation")
