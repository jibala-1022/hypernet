"""
All data handling methods.
"""

from typing import Dict, List, Tuple, Type, Union

import numpy as np
import tensorflow as tf

from ml_intuition import enums
from ml_intuition.data.transforms import BaseTransform

SAMPLES_DIM = 0


def create_tf_dataset(batch_size: int,
                      dataset: Dict[str, np.ndarray],
                      transforms: List[Type[BaseTransform]]) -> Tuple[tf.data.Dataset, int]:
    """
    Create and transform datasets that are used in the training, validaton or testing phases.

    :param batch_size: Size of the batch used in either phase,
        it is the size of samples per gradient step.
    :param dataset: Passed dataset as a dictionary of samples and labels.
    :param transforms: List of all transformations. 
    :return: Transformed dataset with its size.
    """
    n_samples = dataset[enums.Dataset.DATA].shape[SAMPLES_DIM]
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataset[enums.Dataset.DATA], dataset[enums.Dataset.LABELS]))
    for f_transform in transforms:
        dataset = dataset.map(f_transform)
    return dataset.batch(batch_size=batch_size, drop_remainder=False)\
        .repeat().prefetch(tf.contrib.data.AUTOTUNE), n_samples


def shuffle_arrays_together(arrays: List[np.ndarray], seed: int = 0):
    """
    Shuffle arbitrary number of arrays together, in-place

    :param arrays: List of np.ndarrays to be shuffled
    :param seed: seed for the random state, defaults to 0
    :raises AssertionError: When provided arrays have different sizes along 
                            first dimension
    """
    assert all(len(array) == len(arrays[0]) for array in arrays)
    for array in arrays:
        random_state = np.random.RandomState(seed)
        random_state.shuffle(array)


def train_val_test_split(data: np.ndarray, labels: np.ndarray,
                         train_size: Union[int, float] = 0.8,
                         val_size: float = 0.1,
                         stratified: bool = True) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train, val and test sets. The size of the training set 
    is set by the train_size parameter. All the remaining samples will be
    treated as a test set

    :param data: Data with the [SAMPLES, ...] dimensions
    :param labels: Vector with corresponding labels
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
    :return: train_x, train_y, val_x, val_y, test_x, test_y
    :raises AssertionError: When wrong type is passed as train_size
    """
    shuffle_arrays_together([data, labels])
    train_indices = _get_set_indices(labels, train_size, stratified)
    val_indices = _get_set_indices(labels[train_indices], val_size)
    test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
    train_indices = np.setdiff1d(train_indices, val_indices)
    return data[train_indices], labels[train_indices], data[val_indices], \
        labels[val_indices], data[test_indices], labels[test_indices]


def _get_set_indices(labels: np.ndarray, size: float = 0.8,
                     stratified: bool = True) -> np.ndarray:
    """
    Extract indices of a subset of specified data according to size and
    stratified parameters.

    :param labels: Vector with corresponding labels
    :param size: If float, should be between 0.0 and 1.0, if stratified = True, it
                    represents percentage of each class to be extracted,
                 If float and stratified = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :return: Indexes of the train set
    :raises TypeError: When wrong type is passed as size
    """
    unique_labels = np.unique(labels)
    label_indices = [np.where(labels == label)[0] for label in unique_labels]
    assert size > 0, "Size argument must be greater than zero"
    if 0.0 < size < 1.0 and stratified is True:
        for idx in range(len(unique_labels)):
            samples_per_label = int(len(label_indices[idx]) * size)
            label_indices[idx] = label_indices[idx][:samples_per_label]
        train_indices = np.concatenate(label_indices, axis=0)
    elif 0.0 < size < 1.0 and stratified is False:
        train_indices = np.arange(int(len(labels) * size))
    elif size >= 1 and stratified is True:
        for label in range(len(unique_labels)):
            label_indices[label] = label_indices[label][:size]
        train_indices = np.concatenate(label_indices, axis=0)
    elif size >= 1 and stratified is False:
        train_indices = np.arange(size)
    return train_indices


def freeze_session(session: tf.Session,
                   keep_var_names: List[str] = None,
                   output_names: List[str] = None,
                   clear_devices: bool = True) -> tf.GraphDef:
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    :param session: The TensorFlow session to be frozen.
    :param keep_var_names: A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    :param output_names: Names of the relevant graph outputs.
    :param clear_devices: Remove the device directives from the graph for better
                          portability.
    :return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(
                keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
    return frozen_graph
