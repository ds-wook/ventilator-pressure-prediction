import hydra
import neptune.new as neptune
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error

from data.dataset import load_dataset
from model.boosting import LightGBMTrainer


@hydra.main(config_path="../config/training/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)
    submission = pd.read_csv(path + cfg.dataset.submit)

    train, test = load_dataset(path, train, test)

    train = train[train["u_out"] < 1].reset_index(drop=True)

    train_x = train[cfg.dataset.feature_names]
    train_y = train[cfg.dataset.target]
    test_x = test[cfg.dataset.feature_names]
    groups = train[cfg.dataset.groups]

    # make experiment tracking
    run = neptune.init(
        project=cfg.experiment.project,
        tags=list(cfg.experiment.tags),
        capture_hardware_metrics=False,
    )

    lgbm_trainer = LightGBMTrainer(config=cfg, run=run, metric=mean_absolute_error)
    lgbm_trainer.train(train_x, train_y, groups)
    lgbm_trainer.save_model()

    lgbm_preds = lgbm_trainer.predict(test_x)
    lgbm_preds = lgbm_trainer.postprocess(train, lgbm_preds)

    # Save test predictions
    submission["pressure"] = lgbm_preds
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
