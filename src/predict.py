import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../config/train/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")

    lstm_preds = pd.read_csv(submit_path + "blah.csv")
    ensemble_preds = pd.read_csv(submit_path + "median_post_ensemble.csv")
    submission.iloc[:, 1:] = (
        cfg.weight.w1 * lstm_preds["pressure"]
        + cfg.weight.w2 * ensemble_preds["pressure"]
    )
    train = pd.read_csv(path + "train.csv")
    pressure_unique = np.array(sorted(train["pressure"].unique()))
    pressure_min = pressure_unique[0]
    pressure_max = pressure_unique[-1]
    pressure_step = pressure_unique[1] - pressure_unique[0]

    submission["pressure"] = (
        np.round((submission.pressure - pressure_min) / pressure_step) * pressure_step
        + pressure_min
    )
    submission["pressure"] = np.clip(submission.pressure, pressure_min, pressure_max)
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
