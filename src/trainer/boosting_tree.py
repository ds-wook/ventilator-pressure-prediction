import gc
import warnings
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import neptune.new as neptune
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
from sklearn.model_selection import GroupKFold

from utils.utils import LoggerFactory

logger = LoggerFactory().getLogger(__name__)
warnings.filterwarnings("ignore")


class ModelResult(NamedTuple):
    oof_preds: np.ndarray
    preds: Optional[np.ndarray]
    models: Dict[str, any]
    scores: Dict[str, float]


class LGBMTrainer:
    def __init__(self, n_fold: int, metric: Callable):
        self.metric = metric
        self.n_fold = n_fold
        self.result = None

    def train(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        groups: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None,
        verbose: Union[int, bool] = False,
    ) -> bool:
        models = dict()
        scores = dict()

        kf = GroupKFold(n_splits=self.n_fold)
        splits = kf.split(train_x, train_y, groups)
        lgb_oof = np.zeros(train_x.shape[0])

        run = neptune.init(
            project="ds-wook/ventilator-pressure", tags=["LightGBM", "GroupKFold"]
        )

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            print(f"Fold-{fold} Start!")
            neptune_callback = NeptuneCallback(run=run, base_namespace=f"fold_{fold}")

            # create dataset
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # model
            model = LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=100,
                eval_metric="mae",
                verbose=verbose,
                callbacks=[neptune_callback],
            )

            models[f"fold_{fold}"] = model
            # validation
            lgb_oof[valid_idx] = model.predict(X_valid)

            score = self.metric(y_valid.values, lgb_oof[valid_idx])
            scores[f"fold_{fold}"] = score
            logger.info(f"fold {fold}: {score}")

            gc.collect()
            # Log summary metadata to the same run under the "lgbm_summary" namespace
            run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
                booster=model,
                y_pred=lgb_oof[valid_idx],
                y_true=y_valid,
            )

        oof_score = self.metric(train_y.values, lgb_oof)
        logger.info(f"oof score: {oof_score}")

        self.result = ModelResult(
            oof_preds=lgb_oof,
            models=models,
            preds=None,
            scores={
                "oof_score": oof_score,
                "KFoldsScores": scores,
            },
        )
        return True

    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        folds = self.n_fold
        preds = []

        for fold in range(1, folds + 1):
            model = self.result.models[f"fold_{fold}"]
            preds.append(model.predict(test_x))

        lgbm_preds = np.median(np.vstack(preds), axis=0)

        return lgbm_preds

    def postprocess(self, train: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
        all_pressure = np.sort(train.pressure.unique())
        print("The first 25 unique pressures...")
        pressure_min = all_pressure[0].item()
        pressure_max = all_pressure[-1].item()
        pressure_step = (all_pressure[1] - all_pressure[0]).item()

        # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
        preds = (
            np.round((preds - pressure_min) / pressure_step) * pressure_step
            + pressure_min
        )
        preds = np.clip(preds, pressure_min, pressure_max)

        return preds
