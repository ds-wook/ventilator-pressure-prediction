from functools import partial

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import bilstm_data
from tuning.bayesian import BayesianOptimizer, lgbm_objective
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/optimization/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)

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
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    columns = [
        col for col in train.columns if col not in ["id", "breath_id", "pressure"]
    ]
    train_x = train[columns]
    train_y = train["pressure"]
    groups = train["breath_id"]

    objective = partial(
        lgbm_objective, X=train_x, y=train_y, groups=groups, n_fold=cfg.model.fold
    )
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.lgbm_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
