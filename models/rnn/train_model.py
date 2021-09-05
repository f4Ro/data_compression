# ==================================================================================================
# Imports etc.
# ==================================================================================================
from typing import Any
from utils.plotter import Plotter
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from models.shared_code.custom_callback import CustomCallback
# ==================================================================================================
# Data loading, preprocessing and plotting
# ==================================================================================================


def train_model(model: Model, x_train: Any, x_test: Any, num_epochs: int, batch_size: int, plotter: Plotter):
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    own_callback = CustomCallback(
        num_epochs,
        plotter=plotter,
        batch_size=batch_size,
        training_data=x_train,
        validation_data=x_test,
    )

    history = model.fit(
        x_train,
        x_train,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(x_test, x_test),
        callbacks=[own_callback, early_stopping],
        verbose=0,
    ).history
    return history
# ==================================================================================================
# EOF
# ==================================================================================================
