from typing import Optional

import pandas as pd
from neptune.new import Run
from omegaconf import DictConfig, open_dict
from optuna.integration import LightGBMPruningCallback
from optuna.trial import FrozenTrial
from sklearn.metrics import mean_absolute_error

from model.boosting import LightGBMTrainer
from tuning.base import BaseTuner


class LightGBMTuner(BaseTuner):
    def __init__(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        config: DictConfig,
        run: Optional[Run] = None,
    ):
        self.train_x = train_x
        self.train_y = train_y
        super().__init__(config, run)

    def _objective(self, trial: FrozenTrial, config: DictConfig) -> float:
        """
        Objective function
        Args:
            trial: trial object
            config: config object
        Returns:
            metric score
        """
        # trial parameters
        params = {
            "max_depth": trial.suggest_int("max_depth", *config.search.max_depth),
            "subsample": trial.suggest_float("subsample", *config.search.subsample),
            "gamma": trial.suggest_float("gamma", *config.search.gamma),
            "reg_alpha": trial.suggest_float("reg_alpha", *config.search.reg_alpha),
            "reg_lambda": trial.suggest_float("reg_lambda", *config.search.reg_lambda),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *config.search.colsample_bytree
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *config.search.min_child_weight
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *config.search.learning_rate
            ),
        }

        # config update
        with open_dict(config.model):
            config.model.params.update(params)

        # search parameters
        pruning_callback = LightGBMPruningCallback(trial, "l1", valid_name="valid_1")

        lgbm_trainer = LightGBMTrainer(
            run=pruning_callback, search=True, config=config, metric=mean_absolute_error
        )
        result = lgbm_trainer.train(self.train_x, self.train_y)
        score = mean_absolute_error(self.train_y.to_numpy(), result.oof_preds)

        return score
