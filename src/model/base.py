import gc
import logging
import pickle
from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from numpy.typing import ArrayLike
from omegaconf import DictConfig
from pandas import DataFrame, Series
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


@dataclass
class ModelResult:
    """
    Save model's training results
    Args:
        oof_preds: Out-of-fold predictions
        preds: Final predictions
        models: Dictionary of trained models
        scores: Dictionary of scores
    """

    oof_preds: np.ndarray
    preds: Optional[np.ndarray]
    models: Dict[str, any]
    scores: Dict[str, float]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig, metric: Callable, search: bool = False):
        self.config = config
        self.metric = metric
        self.search = search
        self.result = None

    @abstractclassmethod
    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
    ):
        raise NotImplementedError

    def save_model(self):
        """
        Save model
        """
        model_path = Path(get_original_cwd()) / self.config.model.path

        with open(model_path, "wb") as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

    def train(self, train_x: DataFrame, train_y: Series, groups: Series) -> ModelResult:
        """
        Train data
            Parameter:
                train_x: train dataset
                train_y: target dataset
            Return:
                True: Finish Training
        """

        models = dict()
        scores = dict()

        kf = GroupKFold(n_splits=self.config.model.n_splits)
        splits = kf.split(train_x, train_y, groups)

        oof_preds = np.zeros(train_x.shape[0])

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # model
            model = self._train(
                X_train,
                y_train,
                X_valid,
                y_valid,
                fold=fold,
            )
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = model.predict(X_valid)

            score = self.metric(y_valid.to_numpy(), oof_preds[valid_idx])
            scores[f"fold_{fold}"] = score

            if not self.search:
                logging.info(f"Fold {fold}: {score}")

            gc.collect()

            del X_train, X_valid, y_train, y_valid

        oof_score = self.metric(train_y.to_numpy(), oof_preds)

        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            preds=None,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )

        return self.result


def load_model(model_name: str) -> ModelResult:
    """
    Load model
    Args:
        model_name: model name
    Returns:
        ModelResult object
    """
    model_path = Path(get_original_cwd()) / model_name

    with open(model_path, "rb") as output:
        model_result = pickle.load(output)

    return model_result


def predict(result: ModelResult, test_x: DataFrame) -> ArrayLike:
    """
    Predict data
        Parameter:
            test_x: test dataset
        Return:
            preds: inference prediction
    """
    folds = len(result.models)
    preds = []

    for fold in tqdm(range(1, folds + 1)):
        model = result.models[f"fold_{fold}"]
        preds.append(model.predict(test_x))

    lgbm_preds = np.median(np.vstack(preds), axis=0)
    assert len(lgbm_preds) == len(test_x)

    return lgbm_preds


def postprocess(train: DataFrame, preds: ArrayLike) -> ArrayLike:
    """
    Postprocess data
        Parameter:
            train: train dataset
            preds: inference prediction
        Return:
            preds: median prediction
    """
    all_pressure = np.sort(train.pressure.unique())
    print("The first 25 unique pressures...")
    pressure_min = all_pressure[0].item()
    pressure_max = all_pressure[-1].item()
    pressure_step = (all_pressure[1] - all_pressure[0]).item()

    # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
    preds = (
        np.round((preds - pressure_min) / pressure_step) * pressure_step + pressure_min
    )
    preds = np.clip(preds, pressure_min, pressure_max)

    return preds
