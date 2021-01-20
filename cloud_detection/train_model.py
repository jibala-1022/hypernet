""" Train the models for cloud detection. """

import argparse
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path
from tensorflow import keras

from data_gen import DG_38Cloud, load_image_paths
from models import unet
from losses import Jaccard_index_loss, Jaccard_index_metric, Dice_coef_metric, recall, precision, specificity, f1_score
from validate import make_validation_insights

def train_model(dpath: Path, rpath: Path, train_size: float, batch_size: int,
                balance_train_dataset: bool, balance_val_dataset: bool, bn_momentum: float,
                learning_rate: float, stopping_patience: int, epochs: int) -> keras.Model:
    """
    Train the U-Net model using 38-Cloud dataset.

    :param dpath: path to dataset.
    :param rpath: path to direcotry where results and artifacts should be logged.
    :param train_size: proportion of the training set (the rest goes to validation set).
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param balance_train_dataset: whether to balance train dataset.
    :param balance_val_dataset: whether to balance val dataset.
    :param bn_momentum: momentum of the batch normalization layer.
    :param learning_rate: learning rate for training.
    :param stopping_patience: patience param for early stopping.
    :param epochs: number of epochs.
    :return: trained model.
    """
    Path(rpath).mkdir(parents=True, exist_ok=False)
    # Load data
    train_files, val_files = load_image_paths(
        dpath,
        (train_size, 1-train_size)
        )
    # Upstream snow balancing
    traingen = DG_38Cloud(
        files=train_files,
        batch_size=batch_size,
        balance_classes=balance_train_dataset,
        balance_snow=True
        )
    valgen = DG_38Cloud(
        files=val_files,
        batch_size=batch_size,
        balance_classes=balance_val_dataset
        )

    # Create model
    model = unet(input_size=4,
                bn_momentum=bn_momentum
                )
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=Jaccard_index_loss(),
        metrics=[
            keras.metrics.binary_crossentropy,
            keras.metrics.binary_accuracy,
            Jaccard_index_loss(),
            Jaccard_index_metric(),
            Dice_coef_metric(),
            recall,
            precision,
            specificity,
            f1_score
        ]
    )

    # Prepare training
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=stopping_patience,
            verbose=2
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"{rpath}/best_weights",
            save_best_only=True,
            save_weights_only=True,
            verbose=2
        )
    ]

    # Train model
    model.fit_generator(
        generator=traingen,
        epochs=epochs,
        validation_data=valgen,
        callbacks=callbacks,
        verbose=2
        )
    print("Finished fitting. Will make validation insights now.", flush=True)

    # Load best weights
    model.load_weights(f"{rpath}/best_weights")

    # Save validation insights
    best_thr = make_validation_insights(model, valgen, rpath/"validation_insight")

    # Return model
    return model, best_thr


if __name__ == "__main__":
    params = {
        "dpath": Path("../datasets/clouds/38-Cloud/38-Cloud_training"),
        "train_size": 0.8,
        "batch_size": 8,
        "learning_rate": .01,
        "bn_momentum": .9,
        "epochs": 200,
        "stopping_patience": 20,
        }
    train_model(**params)
