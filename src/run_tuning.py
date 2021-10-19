from functools import partial

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import load_dataset
from tuning.bayesian import BayesianOptimizer, lgbm_objective


@hydra.main(config_path="../config/optimization/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)

    train, test = load_dataset(path, train, test)
    train_x = train[cfg.dataset.feature_names]
    train_y = train[cfg.dataset.target]
    groups = train[cfg.dataset.groups]

    objective = partial(
        lgbm_objective, X=train_x, y=train_y, groups=groups, n_fold=cfg.model.fold
    )
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.lgbm_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
