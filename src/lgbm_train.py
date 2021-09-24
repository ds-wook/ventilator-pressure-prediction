import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import load_dataset
from model.gbdt import train_group_kfold_lightgbm


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"

    train = pd.read_csv(path + cfg.dataset.train)
    submission = pd.read_csv(path + cfg.dataset.submit)
    group = train["breath_id"]
    train_x, train_y, test_x = load_dataset(path)

    lgb_preds = train_group_kfold_lightgbm(
        cfg.model.fold,
        train_x,
        train_y,
        test_x,
        group,
        dict(cfg.params),
        cfg.model.verbose,
    )

    # Save test predictions
    submission["pressure"] = lgb_preds
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
