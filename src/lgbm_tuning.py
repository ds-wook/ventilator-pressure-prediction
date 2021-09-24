from functools import partial

import hydra
from omegaconf import DictConfig

from data.dataset import load_dataset
from tuning.bayesian import BayesianOptimizer, lgbm_objective
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/optimization/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = hydra.utils.to_absolute_path(cfg.dataset.path) + "/"
    train, test = load_dataset(path)
    train = reduce_mem_usage(train)
    columns = [
        col for col in train.columns if col not in ["id", "breath_id", "pressure"]
    ]
    train_x = train[columns]
    train_y = train["pressure"]

    objective = partial(lgbm_objective, X=train_x, y=train_y, n_fold=cfg.model.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.lgbm_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
