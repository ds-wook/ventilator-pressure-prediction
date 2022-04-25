import warnings
from typing import Any, Callable, Dict, Optional

import neptune.new.integrations.lightgbm as nep_lgbm_utils
import pandas as pd
from lightgbm import LGBMRegressor
from neptune.new import Run
from neptune.new.integrations.lightgbm import create_booster_summary

from model.base import BaseModel

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(
        self,
        run: Optional[Run],
        config: Dict[str, Any],
        metric: Callable,
        search: bool = False,
    ):
        self.run = run
        super().__init__(config, metric, search)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
    ) -> LGBMRegressor:
        """method train"""

        neptune_callback = (
            nep_lgbm_utils.NeptuneCallback(run=self.run, base_namespace=f"fold_{fold}")
            if not self.search
            else self.run
        )

        model = LGBMRegressor(
            random_state=self.config.model.seed, **self.config.model.params
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            eval_metric=self.config.model.eval_metric,
            verbose=self.config.model.verbose,
            callbacks=[neptune_callback],
        )

        if not self.search:
            # Log summary metadata to the same run under the "lgbm_summary" namespace
            self.run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
                booster=model,
                y_pred=model.predict(X_valid),
                y_true=y_valid,
            )

        return model
