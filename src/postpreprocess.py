import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../config/train/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"

    train = pd.read_csv(path + "train.csv")
    ensemble_preds = pd.read_csv(submit_path + "rwb_121_loops.csv")

    pressure_unique = np.array(sorted(train["pressure"].unique()))
    ensemble_preds["pressure"] = ensemble_preds["pressure"].map(
        lambda x: pressure_unique[np.abs(pressure_unique - x).argmin()]
    )

    ensemble_preds.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
