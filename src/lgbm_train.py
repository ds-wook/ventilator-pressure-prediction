import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import load_dataset
from model.gbdt import train_group_kfold_lightgbm
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)
    submission = pd.read_csv(path + cfg.dataset.submit)

    train = load_dataset(train)
    test = load_dataset(test)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    lgb_preds = train_group_kfold_lightgbm(
        cfg.model.fold,
        train,
        test,
        dict(cfg.params),
        cfg.model.verbose,
    )

    # Save test predictions
    submission["pressure"] = lgb_preds
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
