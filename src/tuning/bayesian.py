import warnings
from typing import Callable, Sequence, Union

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import numpy as np
import optuna
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from lightgbm import LGBMRegressor
from neptune.new.exceptions import NeptuneMissingApiTokenException
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import mean_absolute_error

from model.model_selection import ShufflableGroupKFold

warnings.filterwarnings("ignore")


class BayesianOptimizer:
    def __init__(
        self, objective_function: Callable[[Trial], Union[float, Sequence[float]]]
    ):
        self.objective_function = objective_function

    def build_study(self, trials: FrozenTrial, verbose: bool = False):
        try:
            run = neptune.init(
                project="ds-wook/optiver-prediction", tags="optimization"
            )

            neptune_callback = optuna_utils.NeptuneCallback(
                run, plots_update_freq=1, log_plot_slice=False, log_plot_contour=False
            )
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="TPE Optimization",
                direction="minimize",
                sampler=sampler,
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            )
            study.optimize(
                self.objective_function, n_trials=trials, callbacks=[neptune_callback]
            )
            run.stop()

        except NeptuneMissingApiTokenException:
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="optimization", direction="minimize", sampler=sampler
            )
            study.optimize(self.objective_function, n_trials=trials)
        if verbose:
            self.display_study_statistics(study)

        return study

    @staticmethod
    def display_study_statistics(study: Study):
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    '{key}': {value},")

    @staticmethod
    def lgbm_save_params(study: Study, params_name: str):
        params = study.best_trial.params
        params["n_estimators"] = 30000
        params["boosting_type"] = "gbdt"
        params["objective"] = "mae"
        params["n_jobs"] = -1

        with open(to_absolute_path("../../config/train/train.yaml")) as f:
            train_dict = yaml.load(f, Loader=yaml.FullLoader)
        train_dict["params"] = params

        with open(to_absolute_path("../../config/train/" + params_name), "w") as p:
            yaml.dump(train_dict, p)


def lgbm_objective(
    trial: FrozenTrial,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_fold: int,
) -> float:
    params = {
        "n_estimators": 20000,
        "objective": "mae",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 4e-01, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-02, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-02, 1),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1),
        "subsample": trial.suggest_uniform("subsample", 0.1, 1),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
    }
    pruning_callback = LightGBMPruningCallback(trial, "RMSPE", valid_name="valid_1")

    group_kf = ShufflableGroupKFold(n_splits=n_fold, random_state=2021, shuffle=True)
    splits = group_kf.split(X, y, groups)
    lgbm_oof = np.zeros(X.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # model
        model = LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric="mae",
            early_stopping_rounds=100,
            verbose=False,
            callbacks=[pruning_callback],
        )
        # validation
        lgbm_oof[valid_idx] = model.predict(
            X_valid, num_iteration=model.best_iteration_
        )

    score = mean_absolute_error(y, lgbm_oof)
    return score
