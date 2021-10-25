from models.cbn_vae.create_model import create_model as create_cbn_vae
from models.rnn.create_model import create_model as create_rnn

configurations = {
    "cbn_vae": {
        "num_epochs": 20,
        "epoch_interval": 2,
        "train_test_cutoff": 320,
        "batch_size": 32,
        "sequence_length": 120,
        "create_model": create_cbn_vae,
        "default_config": {
            'encoder_activation': 'tanh',
            'decoder_activation': 'tanh',
            'dense_nodes': 33,
            'bottleneck_activation': 'relu',
            'lr': 0.001521356711612709,
            'optimizer': 'Adam',
            'sgd_momentum': 0.5091287212784572
        }
    },
    "rnn": {
        "num_epochs": 10,
        "epoch_interval": 1,
        "train_test_cutoff": 1900,
        "batch_size": 1,
        "sequence_length": 20,
        "create_model": create_rnn,
        "default_config": {

        }
    }
}
