import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import add_features, bilstm_data
from model.gbdt import train_group_kfold_lightgbm
from utils.utils import find_nearest, reduce_mem_usage


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)
    submission = pd.read_csv(path + cfg.dataset.submit)
    train_bilstm = pd.read_csv(path + "lstm_train.csv")
    test_bilstm = pd.read_csv(path + "lstm_test.csv")

    pressure_unique = np.array(sorted(train["pressure"].unique()))
    train = pd.merge(train, train_bilstm, on="id")
    test = pd.merge(test, test_bilstm, on="id")

    train = bilstm_data(train, pressure_unique, cfg.dataset.num)
    test = bilstm_data(test, pressure_unique, cfg.dataset.num)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    train = add_features(train)
    test = add_features(test)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    lgbm_preds = train_group_kfold_lightgbm(
        cfg.model.fold,
        train,
        test,
        dict(cfg.params),
        cfg.model.verbose,
    )

    all_pressure = np.sort(train.pressure.unique())
    print("The first 25 unique pressures...")

    # Save test predictions
    submission["pressure"] = lgbm_preds

    # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
    submission["pressure"] = submission["pressure"].apply(
        lambda x: find_nearest(all_pressure, x)
    )
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
