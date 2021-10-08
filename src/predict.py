import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from utils.utils import find_nearest


@hydra.main(config_path="../config/train/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + "train.csv")
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")

    lstm_preds = pd.read_csv(submit_path + "rwb-125-loops.csv")
    ensemble_preds = pd.read_csv(submit_path + "add_stacking_lightgbm.csv")
    submission["pressure"] = (
        cfg.weight.w1 * lstm_preds["pressure"]
        + cfg.weight.w2 * ensemble_preds["pressure"]
    )

    all_pressure = np.sort(train.pressure.unique())
    print("The first 25 unique pressures...")

    # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
    submission["pressure"] = submission["pressure"].apply(
        lambda x: find_nearest(all_pressure, x)
    )
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
