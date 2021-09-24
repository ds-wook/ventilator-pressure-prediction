import warnings
from typing import Any, Dict, Optional, Tuple, Union

import neptune.new as neptune
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
from sklearn.metrics import mean_absolute_error

from model.model_selection import ShufflableGroupKFold
from utils.utils import timer

warnings.filterwarnings("ignore")


def train_group_kfold_lightgbm(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    kf = ShufflableGroupKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y, groups)
    lgb_oof = np.zeros(X.shape[0])
    lgb_preds = np.zeros(X_test.shape[0])

    run = neptune.init(
        project="ds-wook/ventilator-pressure", tags=["LightGBM", "GroupKFold"]
    )

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        neptune_callback = NeptuneCallback(run=run, base_namespace=f"fold_{fold}")
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        with timer(f"Fold-{fold} Start"):
            # model
            model = LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=100,
                eval_metric="mae",
                categorical_feature=["R", "C"],
                verbose=verbose,
                callbacks=[neptune_callback],
            )
        # validation
        lgb_oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration_)
        lgb_preds += model.predict(X_test, num_iteration=model.best_iteration_) / n_fold

        # Log summary metadata to the same run under the "lgbm_summary" namespace
        run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
            booster=model,
            log_trees=True,
            list_trees=[0, 1, 2, 3, 4],
            max_num_features=20,
            y_pred=lgb_oof[valid_idx],
            y_true=y_valid,
        )

    print(f"Total Performance MAE: {mean_absolute_error(y, lgb_oof)}")
    run.stop()

    return lgb_preds
