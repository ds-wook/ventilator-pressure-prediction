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
    submission = pd.read_csv(path + cfg.dataset.submit)

    train, test = load_dataset(path)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    print(train.shape)

    columns = [
        col for col in train.columns if col not in ["id", "breath_id", "pressure"]
    ]

    train_x = train[columns]
    train_y = train["pressure"]
    test_x = test[columns]
    groups = train["breath_id"]

    lgb_preds = train_group_kfold_lightgbm(
        cfg.model.fold,
        train_x,
        train_y,
        test_x,
        groups,
        dict(cfg.params),
        cfg.model.verbose,
    )

    # Save test predictions
    submission["pressure"] = lgb_preds
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
