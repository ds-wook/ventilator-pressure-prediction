import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import add_features, bilstm_data
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"

    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)
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
    test[cfg.dataset.target] = preds[cfg.dataset.target]
    train = pd.concat([train, test], axis=0)
    train = reduce_mem_usage(train)
    train.to_csv(path + "psudo_train.csv", index=False)
    test.to_csv(path + "final_test.csv", index=False)


if __name__ == "__main__":
    _main()
