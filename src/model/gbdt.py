import warnings
from typing import Any, Dict, Optional, Tuple, Union

import neptune.new as neptune
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


def train_group_kfold_lightgbm(
    n_fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    columns = [
        col for col in train.columns if col not in ["id", "breath_id", "pressure"]
    ]
    train_x = train[columns]
    train_y = train["pressure"]
    test_x = test[columns]
    groups = train["breath_id"]
    kf = GroupKFold(n_splits=n_fold)
    splits = kf.split(train, train_y, groups)
    lgb_oof = np.zeros(train_x.shape[0])
    lgb_preds = np.zeros(test_x.shape[0])

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
        model = LGBMRegressor(**params, n_estimators=10000)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=15,
            eval_metric="mae",
            verbose=verbose,
            callbacks=[neptune_callback],
        )
        # validation
        lgb_oof[valid_idx] = model.predict(X_valid)
        lgb_preds += model.predict(test_x) / n_fold

        # Log summary metadata to the same run under the "lgbm_summary" namespace
        run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
            booster=model,
            log_trees=True,
            list_trees=[0, 1, 2, 3],
            y_pred=lgb_oof[valid_idx],
            y_true=y_valid,
        )

    print(f"Total Performance MAE: {mean_absolute_error(train_y, lgb_oof)}")
    run.stop()

    return lgb_preds
