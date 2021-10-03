import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import bilstm_data, add_features
from model.gbdt import train_group_kfold_lightgbm
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)
    submission = pd.read_csv(path + cfg.dataset.submit)
    train_bilstm = pd.read_csv(path + "lstm_train.csv")
    test_bilstm = pd.read_csv(path + "lstm_test.csv")

    train = pd.merge(train, train_bilstm, on="id")
    test = pd.merge(test, test_bilstm, on="id")
    train.rename(
        columns={f"pressure{i}": f"bilstm_pred{i}" for i in range(10)}, inplace=True
    )
    test.rename(
        columns={f"pressure{i}": f"bilstm_pred{i}" for i in range(10)}, inplace=True
    )
    train = bilstm_data(train)
    test = bilstm_data(test)
    train = add_features(train)
    test = add_features(test)

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
