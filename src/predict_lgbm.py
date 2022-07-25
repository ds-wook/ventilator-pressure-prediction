import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import load_test_dataset
from model.base import load_model, postprocess, predict


@hydra.main(config_path="../config/training/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + cfg.dataset.submit)
    train = pd.read_csv(path + cfg.dataset.train)
    test_x = load_test_dataset(cfg)

    lgbm_trainer = load_model(cfg.model.lightgbm)

    lgbm_preds = predict(lgbm_trainer, test_x)
    lgbm_preds = postprocess(train, lgbm_preds)

    # Save test predictions
    submission["pressure"] = lgbm_preds
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
