import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


def get_memory(model: Model, batch_size: int = 1) -> int:
    """
    Get the memory that is required for the model to work.


    Get the number of activations that have to be stored.
    Through one forward pass the activations are stored in memory, thus
    calculate the size of each layers outputs and multiply them by the memory
    occupied by a single number.

    Get the number of parameters that the model consists of.
    As each parameter is one number (which is usually a float32 -> 4 bytes), the
    memory occupied by the model can be obtained by the number of parameters
    directly

    These measurements are always assumed to be performed for a batch size of 1
    (as an increasing batch size will increase the resource consumption and is
    therefore worse in a constrained environment)
    """
    # !Calculate the size of the activations
    shapes_mem_count = 0           # Counts the number of activations that are stored
    internal_model_mem_count = 0   # Counts the number of parameters for any sub-models
    for layer in model.layers:
        # If the model consists of multiple smaller models, perform the function
        # recursively for each of them
        layer_type = layer.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_memory(batch_size, layer)

        # Go through each of the dimensions of the output shape, multiply them
        # together to obtain the total volume of the layer and add it to the
        # total count
        single_layer_mem = 1
        out_shape = layer.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    # !Calculate the size of the parameters
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    # Determine the size of a single number, as numbers are usualy float32, this
    # defaults to 4 bytes (also 2 and 8 bytes are possbile for float16 and
    # float64 respectively)
    single_number_memory_size = 4.0
    if K.floatx() == 'float16':
        single_number_memory_size = 2.0
    if K.floatx() == 'float64':
        single_number_memory_size = 8.0

    total_memory = single_number_memory_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    total_memory += internal_model_mem_count
    return total_memory
