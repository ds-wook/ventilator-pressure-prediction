import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../config/train/", config_name="postprocess.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")
    lgbm_preds = pd.read_csv(submit_path + cfg.dataset.lightgbm)
    lstm1_preds = pd.read_csv(submit_path + cfg.dataset.lstm1)
    lstm2_preds = pd.read_csv(submit_path + cfg.dataset.lstm2)
    ensemble_preds = pd.read_csv(submit_path + cfg.dataset.ensemble)
    blending_preds = np.median(
        [
            lgbm_preds.pressure.values,
            lstm1_preds.pressure.values,
            lstm2_preds.pressure.values,
            ensemble_preds.pressure.values,
        ]
    )

    submission["pressure"] = blending_preds
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
