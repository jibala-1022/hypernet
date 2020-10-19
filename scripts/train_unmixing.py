"""
Perform the training of the model for the unmixing problem.
"""

import os
from typing import Dict

import numpy as np
import tensorflow as tf

from ml_intuition import enums, models
from ml_intuition.data import io, transforms
from ml_intuition.data.preprocessing import reshape_cube_to_2d_samples, \
    reshape_cube_to_3d_samples
from ml_intuition.data.utils import get_central_pixel_spectrum
from ml_intuition.evaluation import time_metrics
from ml_intuition.evaluation.performance_metrics import \
    spectral_information_divergence_loss, \
    overall_rms_abundance_angle_distance, \
    cnn_rmse, sum_per_class_rmse
from ml_intuition.models import unmixing_cube_based_dcae, \
    unmixing_pixel_based_dcae, \
    unmixing_cube_based_cnn, unmixing_pixel_based_cnn


def train_dcae(model, data, **kwargs):
    """
    Function for running experiments on convolutional neural network (CNN),
    given a set of hyperparameters.
    :param model: The model used for training.
    :param data: Either path to the input data or the data dict itself.
        First dimension of the dataset should be the number of samples.
    :param kwargs: The keyword arguments containing additional hyperparameters
        dependent on the type of the network architecture.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=kwargs['lr']),
        loss=spectral_information_divergence_loss,
        metrics=[spectral_information_divergence_loss])

    time_history = time_metrics.TimeHistory()

    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(kwargs['dest_path'], kwargs['model_name']),
        save_best_only=True,
        monitor='loss',
        mode='min')

    train_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=kwargs['patience'], mode='min')

    data = transforms.apply_transformations(data, [
        transforms.PerBandMinMaxNormalization(
            **transforms.PerBandMinMaxNormalization.get_min_max_vectors(
                (data['data'])))])

    if kwargs['neighborhood_size'] is None:
        data['data'], data['labels'] = reshape_cube_to_2d_samples(
            data['data'], data['labels'], -1, True)

    else:
        data['data'], data['labels'] = reshape_cube_to_3d_samples(
            data['data'], data['labels'],
            kwargs['neighborhood_size'], -1, -1, True)
    data['data'] = np.expand_dims(data['data'], -1)
    history = model.fit(
        x=data['data'],
        y=get_central_pixel_spectrum(data['data'],
                                     kwargs['neighborhood_size']),
        epochs=kwargs['epochs'],
        verbose=kwargs['verbose'],
        shuffle=kwargs['shuffle'],
        callbacks=[time_history, mcp_save, train_early_stopping],
        batch_size=kwargs['batch_size'])

    history.history[time_metrics.TimeHistory.__name__] = \
        time_history.average

    io.save_metrics(dest_path=kwargs['dest_path'],
                    file_name='training_metrics.csv',
                    metrics=history.history)


def train_cnn(model, data, **kwargs):
    """
    Function for running experiments on deep convolutional autoencoder (DCAE),
    given a set of hyperparameters.
    :param model: The model used for training.
    :param data: Either path to the input data or the data dict itself.
        First dimension of the dataset should be the number of samples.
    :param kwargs: The keyword arguments containing additional hyperparameters
        dependent on the type of the network architecture.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=kwargs['lr']),
        loss='mse',
        metrics=[cnn_rmse,
                 overall_rms_abundance_angle_distance,
                 sum_per_class_rmse])

    time_history = time_metrics.TimeHistory()
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(kwargs['dest_path'], kwargs['model_name']),
        save_best_only=True,
        monitor='val_loss',
        mode='min')
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=kwargs['patience'], mode='min')

    callbacks = [time_history, mcp_save, early_stopping]

    train_dict = data[enums.Dataset.TRAIN]
    val_dict = data[enums.Dataset.VAL]

    min_, max_ = data[enums.DataStats.MIN], data[enums.DataStats.MAX]

    transformations = [transforms.SpectralTransform(),
                       transforms.MinMaxNormalize(min_=min_, max_=max_)]

    train_dict = transforms.apply_transformations(train_dict, transformations)
    val_dict = transforms.apply_transformations(val_dict, transformations)

    history = model.fit(
        x=train_dict[enums.Dataset.DATA],
        y=train_dict[enums.Dataset.LABELS],
        epochs=kwargs['epochs'],
        verbose=kwargs['verbose'],
        shuffle=kwargs['shuffle'],
        validation_data=(val_dict[enums.Dataset.DATA],
                         val_dict[enums.Dataset.LABELS]),
        callbacks=callbacks,
        batch_size=kwargs['batch_size'])

    np.savetxt(os.path.join(kwargs['dest_path'],
                            'min-max.csv'), np.array([min_, max_]),
               delimiter=',', fmt='%f')

    history.history[time_metrics.TimeHistory.__name__] = time_history.average

    io.save_metrics(dest_path=kwargs['dest_path'],
                    file_name='training_metrics.csv',
                    metrics=history.history)


TRAIN_FUNCTIONS = {
    unmixing_cube_based_dcae.__name__: train_dcae,
    unmixing_pixel_based_dcae.__name__: train_dcae,

    unmixing_cube_based_cnn.__name__: train_cnn,
    unmixing_pixel_based_cnn.__name__: train_cnn
}


def train(data: Dict[str, np.ndarray],
          model_name: str,
          dest_path: str,
          sample_size: int,
          n_classes: int,
          neighborhood_size: int = None,
          kernel_size: int = 5,
          n_kernels: int = 200,
          n_layers: int = 1,
          lr: float = 0.001,
          batch_size: int = 128,
          epochs: int = 200,
          verbose: int = 2,
          shuffle: bool = True,
          patience: int = 15,
          endmembers_path: str = None,
          seed: int = 0):
    """
    Function for running experiments given a set of hyper parameters.
    :param data: Either path to the input data or the data dict itself.
        First dimension of the dataset should be the number of samples.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param dest_path: Path to where all experiment runs will be saved as
        subdirectories in this directory.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    :param neighborhood_size: Size of the spatial patch.
    :param kernel_size: Size of ech kernel in each layer.
    :param n_kernels: Number of kernels in each layer.
    :param n_layers: Number of layers in the model.
    :param lr: Learning rate for the model, i.e., regulates
        the size of the step in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle dataset
     dataset_key each epoch.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    :param endmembers_path: Path to the endmembers file
        containing average reflectances for each class, i.e., the pure spectra.
        Used only when use_unmixing is true.
    :param seed: Seed for training reproducibility.
    """
    # Reproducibility:
    tf.reset_default_graph()
    tf.set_random_seed(seed=seed)
    np.random.seed(seed=seed)

    model = models.get_model(
        model_key=model_name,
        **{'kernel_size': kernel_size,
           'n_kernels': n_kernels,
           'n_layers': n_layers,
           'input_size': sample_size,
           'n_classes': n_classes,
           'neighborhood_size': neighborhood_size,
           'endmembers': np.load(
               endmembers_path) if endmembers_path is not None else None})

    model.summary()
    # Run specific training function and pass specific hyperparameters.
    TRAIN_FUNCTIONS[model_name](
        model, data,
        **{'dest_path': dest_path,
           'model_name': model_name,
           'batch_size': batch_size,
           'epochs': epochs,
           'shuffle': shuffle,
           'verbose': verbose,
           'lr': lr,
           'patience': patience,
           'neighborhood_size': neighborhood_size})
