"""
Evaluate the dataset using and ensemble for the unmixing problem.
"""
from typing import Dict, List

import numpy as np

from ml_intuition import enums
from ml_intuition.data import io
from ml_intuition.evaluation.performance_metrics import \
    calculate_unmixing_metrics
from ml_intuition.evaluation.time_metrics import timeit
from ml_intuition.models import Ensemble


def evaluate(*,
             y_pred: np.ndarray,
             data: Dict,
             dest_path: str,
             voting: str,
             train_set_predictions: List[np.ndarray] = None,
             voting_model: str = None,
             voting_model_params: str = None):
    """
    Evaluate the ensemble for the unmixing problem.

    :param y_pred: Predictions made on the test set.
    :param data: Either path to the input data or the data dict.
    :param dest_path: Directory in which to store the calculated metrics.
    :param voting: Method of ensemble voting. If 'booster',
        employs a new model, which is trained on the
        ensemble predictions on the training set. Else if 'mean', averages
        the predictions of all models, without any weights.
    :param train_set_predictions: Predictions made on the train set.
        Only used if 'voting' = 'booster'.
   :param voting_model: Type of the model to use when the voting
        argument is set to 'booster'. This indicates, that a new model
        is trained on the ensemble's predictions on the learning set,
        to leverage the quality of the regression. Supported models are:
        SVR (support vector machine for regression), RFR (random forest
        for regression) and DTR (decision tree for regression).
    :param voting_model_params: Parameters of the voting model.
        Used only when the type of voting is set to 'booster'.
        Should be specified analogously to the noise injection parameters
        in the 'noise' module.
    """
    ensemble = Ensemble(voting=voting)
    if voting == 'booster':
        train_set_predictions = np.array(train_set_predictions)
        ensemble.train_ensemble_predictor(
            train_set_predictions,
            data[enums.Dataset.TRAIN][enums.Dataset.LABELS],
            voting_model,
            voting_model_params)
    vote = timeit(ensemble.vote)
    y_pred, voting_time = vote(y_pred)
    model_metrics = calculate_unmixing_metrics(**{
        'y_pred': y_pred,
        'y_true': data[enums.Dataset.TEST][enums.Dataset.LABELS],
        'endmembers': None})
    model_metrics['inference_time'] = [voting_time]
    io.save_metrics(dest_path=dest_path,
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)
