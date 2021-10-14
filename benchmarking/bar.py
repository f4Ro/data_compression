from benchmarking.foo import configurations
from models.shared_code.get_data_and_config import get_data_and_config
from benchmarking.benchmarks import run_benchmarks



def foo():
    config = configurations['cbn_vae']
    create_model = configurations['create_model']

    x_train, x_test, _, _ = get_data_and_config(
        'intel', config['sequence_length'], config['batch_size'], config['train_test_cutoff'])

    encoder, decoder, model = create_model(config['default_config'])

    num_epochs = config['num_epochs']
    # epoch_interval = config['epoch_interval']
    # callback = CustomCallback(
    #     num_epochs=num_epochs, epoch_interval=epoch_interval,
    #     plotter=plotter,
    #     batch_size=batch_size,
    #     training_data=x_train, validation_data=x_test)
    model.fit(
        x_train, x_train,
        batch_size=config['batch_size'],
        epochs=num_epochs,
        validation_data=(x_test, x_test),
        callbacks=[],
        verbose=0
    )
