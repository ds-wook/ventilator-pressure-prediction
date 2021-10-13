import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import add_features, bilstm_data
from trainer.gbdt import train_group_kfold_lightgbm
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"

    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)
    submission = pd.read_csv(path + cfg.dataset.submit)
    train_bilstm = pd.read_csv(path + "lstm_train.csv")
    test_bilstm = pd.read_csv(path + "lstm_test.csv")

    train = pd.merge(train, train_bilstm, on="id")
    test = pd.merge(test, test_bilstm, on="id")

    train = bilstm_data(train, cfg.dataset.num)
    test = bilstm_data(test, cfg.dataset.num)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    train = add_features(train)
    test = add_features(test)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    preds = pd.read_csv(submit_path + "median_stacking_lightgbm.csv")
    test = pd.merge(test, preds, on="id")

    print("Psudo Labeling Start")
    train = pd.concat([train, test], axis=0)
    train = reduce_mem_usage(train)

    lgbm_preds = train_group_kfold_lightgbm(
        cfg.model.fold,
        train,
        test,
        dict(cfg.params),
        cfg.model.verbose,
    )

    all_pressure = np.sort(train.pressure.unique())
    print("The first 25 unique pressures...")
    pressure_min = all_pressure[0].item()
    pressure_max = all_pressure[-1].item()
    pressure_step = (all_pressure[1] - all_pressure[0]).item()

    # Save test predictions
    submission["pressure"] = lgbm_preds

    # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
    submission["pressure"] = (
        np.round((submission.pressure - pressure_min) / pressure_step) * pressure_step
        + pressure_min
    )
    submission.pressure = np.clip(submission.pressure, pressure_min, pressure_max)
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
