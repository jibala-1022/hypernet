"""
All models that are used for training.
"""

import sys

import tensorflow as tf


def model_2d(kernel_size: int,
             n_kernels: int,
             n_layers: int,
             input_size: int,
             n_classes: int,
             **kwargs) -> tf.keras.Sequential:
    """
    2D model which consists of 2D convolutional blocks.

    :param kernel_size: Size of the convolutional kernel.
    :param n_kernels: Number of kernels, i.e., the activation maps in each layer.
    :param n_layers: Number of layers in the network.
    :param input_size: Number of input channels, i.e., the number of spectral bands.
    :param n_classes: Number of classes.
    """

    def add_layer(model):
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(3, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(2, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1), strides=(2, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        return model
    model = tf.keras.Sequential()

    for _ in range(n_layers):
        model = add_layer(model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=200, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def pool_model_2d(kernel_size: int,
                  n_kernels: int,
                  n_layers: int,
                  input_size: int,
                  n_classes: int,
                  **kwargs) -> tf.keras.Sequential:
    """
    2D model which consists of 2D convolutional layers and 2D pooling layers.

    :param kernel_size: Size of the convolutional kernel.
    :param n_kernels: Number of kernels, i.e., the activation maps in each layer.
    :param n_layers: Number of layers in the network.
    :param input_size: Number of input channels, i.e., the number of spectral bands.
    :param n_classes: Number of classes.
    """

    def add_layer(model):
        model.add(tf.keras.layers.Conv2D(n_kernels, (kernel_size, 1),
                                         strides=(2, 1),
                                         input_shape=(input_size, 1, 1),
                                         padding="valid",
                                         activation='relu'))
        model.add(tf.keras.layers.BatchNormalization()),
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(1, 1)))
        return model
    model = tf.keras.Sequential()

    for _ in range(n_layers):
        model = add_layer(model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def get_model(model_key: str, **kwargs):
    """
    Get a given instance of model specified by mode_key.

    :param model_key: Specifies which model to use.
    :param kwargs: Any keyword arguments that the model accepts.
    """
    # Get the list of all model creating functions and their name as the key:
    all_ = {
        str(f): eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](**kwargs)


def model_3d_mfl(kernel_size: int,
                 n_kernels: int,
                 n_classes: int,
                 input_size: int,
                 **kwargs):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=n_kernels,
                                     kernel_size=kernel_size - 3,
                                     strides=(1, 1),
                                     input_shape=(kernel_size, kernel_size,
                                                  input_size),
                                     data_format='channels_last',
                                     padding='valid'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=n_kernels,
                                     kernel_size=(2, 2),
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=n_classes,
                                     kernel_size=(2, 2),
                                     padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Softmax())
    return model


def model_3d_deep(n_classes: int, input_size: int, **kwargs):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu', input_shape=(7, 7, input_size, 1), data_format='channels_last'))
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model
