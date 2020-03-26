"""
Run experiments given set of hyperparameters.
"""

import os

import clize
import tensorflow as tf
from scripts import evaluate_model, prepare_data, train_model

EXPERIMENT = 'experiment'


def run_experiments(*,
                    data_file_path: str,
                    ground_truth_path: str,
                    train_size: float = 0.8,
                    val_size: float = 0.1,
                    stratified: bool = True,
                    background_label: int = 0,
                    channels_idx: int = 0,
                    save_data: bool = False,
                    n_runs: int,
                    model_name: str,
                    kernel_size: int = 3,
                    n_kernels: int = 16,
                    n_layers: int = 1,
                    dest_path: str,
                    sample_size: int,
                    n_classes: int,
                    lr: float = 0.005,
                    batch_size: int = 150,
                    epochs: int = 10,
                    verbose: int = 1,
                    shuffle: bool = True,
                    patience: int = 3):
    """
    Function for running experiments given a set of hyperparameters.
    :param data_file_path: Path to the data file. Supported types are: .npy
    :param ground_truth_path: Path to the ground-truth data file.
    :param train_size: If float, should be between 0.0 and 1.0,
                        if stratified = True, it represents percentage of each
                        class to be extracted,
                 If float and stratified = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage of
                     each class from the training set to be extracted as a
                     validation set, defaults to 0.1
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :param background_label: Label indicating the background in GT file
    :param channels_idx: Index specifying the channels position in the provided
                         data
    :param save_data: Whether to save the prepared dataset
    :param n_runs: Number of total experiment runs.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param kernel_size: Size of ech kernel in each layer.
    :param n_kernels: Number of kernels in each layer.
    :param n_layers: Number of layers in the model.
    :param dest_path: Path to where all experiment runs will be saved as subfolders
        in this directory.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    :param lr: Learning rate for the model, i.e., regulates the size of the step
        in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle dataset
     dataset_key each epoch.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    """
    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(
            dest_path, '{}_{}'.format(EXPERIMENT, str(experiment_id)))
        if save_data:
            data_source = os.path.join(experiment_dest_path, 'data.h5')
        else:
            data_source = None

        os.makedirs(experiment_dest_path, exist_ok=True)

        data = prepare_data.main(data_file_path=data_file_path,
                                 ground_truth_path=ground_truth_path,
                                 output_path=data_source,
                                 train_size=train_size,
                                 val_size=val_size,
                                 stratified=stratified,
                                 background_label=background_label,
                                 channels_idx=channels_idx,
                                 save_data=save_data,
                                 seed=experiment_id)
        if not save_data:
            data_source = data

        train_model.train(model_name=model_name,
                          kernel_size=kernel_size,
                          n_kernels=n_kernels,
                          n_layers=n_layers,
                          dest_path=experiment_dest_path,
                          data=data_source,
                          sample_size=sample_size,
                          n_classes=n_classes,
                          lr=lr,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          patience=patience)

        evaluate_model.evaluate(
            model_path=os.path.join(experiment_dest_path, model_name),
            data=data_source,
            verbose=verbose,
            n_classes=n_classes)

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    clize.run(run_experiments)
