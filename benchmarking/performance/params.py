from tensorflow.keras.models import Model


def get_params(model: Model) -> int:
    """
    Get the number of parameters for a given keras model
    """
    return model.count_params()
