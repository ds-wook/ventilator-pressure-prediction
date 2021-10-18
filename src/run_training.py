import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error

from data.dataset import load_dataset
from trainer.boosting_tree import LightGBMTrainer


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)
    submission = pd.read_csv(path + cfg.dataset.submit)
    model_name = list(cfg.model)[0]
    train, test = load_dataset(path, train, test, cfg.dataset.num)
    
    train_x = train[cfg.dataset.feature_names]
    train_y = train[cfg.dataset.target]
    test_x = test[cfg.dataset.feature_names]
    groups = train[cfg.dataset.groups]

    if model_name == "lightgbm":
        lgbm_trainer = LightGBMTrainer(cfg.model.fold, mean_absolute_error)
        lgbm_trainer.train(
            train_x, train_y, groups, cfg.model.lightgbm.params, cfg.model.verbose
        )
        lgbm_preds = lgbm_trainer.predict(test_x)
        lgbm_preds = lgbm_trainer.postprocess(train, lgbm_preds)
        lgbm_oof = lgbm_trainer.result.oof_preds

        # Save train predictions
        train["lgbm_preds"] = lgbm_oof
        train[["id", "lgbm_preds"]].to_csv(path + "stacking_oof.csv", index=False)
        # Save test predictions
        submission["pressure"] = lgbm_preds
        submission.to_csv(submit_path + cfg.submit.name, index=False)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
