import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../config/train/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")

    lstm_preds = pd.read_csv(submit_path + "lstm_5fold.csv")
    lgbm_preds = pd.read_csv(submit_path + "feg_lightgbm_5fold.csv")
    lgbm_preds.to_csv(submit_path + "lightgbm_post_preprocessing.csv", index=False)
    submission.iloc[:, 1:] = (
        cfg.weight.w1 * lstm_preds["pressure"]
        + cfg.weight.w2 * lgbm_preds["pressure"]
    )

    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
