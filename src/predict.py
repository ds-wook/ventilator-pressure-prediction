import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../config/train/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + "train.csv")
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")

    lstm1_preds = pd.read_csv(submit_path + "ventilator_lstm_model.csv")
    lgbm_preds = pd.read_csv(submit_path + "median_stacking_lightgbm.csv")
    lstm2_preds = pd.read_csv(submit_path + "median_fine_tune_lstm.csv")
    submission["pressure"] = (
        cfg.weight.w1 * lstm1_preds["pressure"]
        + cfg.weight.w2 * lgbm_preds["pressure"]
        + cfg.weight.w3 * lstm2_preds["pressure"]
    )

    print("Postprocessing!")
    pressure_unique = np.array(sorted(train["pressure"].unique()))
    submission["pressure"] = submission["pressure"].map(
        lambda x: pressure_unique[np.abs(pressure_unique - x).argmin()]
    )

    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
