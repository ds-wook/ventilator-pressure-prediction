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
    all_pressure = np.sort(train.pressure.unique())

    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")
    lgbm_preds = pd.read_csv(submit_path + cfg.dataset.lightgbm)
    # lstm1_preds = pd.read_csv(submit_path + cfg.dataset.lstm1)
    ensemble_preds = pd.read_csv(submit_path + cfg.dataset.ensemble)
    blending_preds = (
        cfg.weight.w1 * ensemble_preds.pressure + cfg.weight.w1 * lgbm_preds.pressure
    )
    print("Postprocess")
    submission["pressure"] = blending_preds
    submission["pressure"] = submission["pressure"].map(
        lambda x: find_nearest(all_pressure, x)
    )
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
