import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import load_dataset
from model.gbdt import train_group_kfold_lightgbm
from utils.utils import timer


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"

    with timer("LightGBM Learning"):
        train, test = load_dataset(path)

        # Split features and target
        X = train.drop(["breath_id", "pressure"], axis=1)
        y = train["pressure"]
        X_test = test.drop(["breath_id"], axis=1)
        group = train["breath_id"]

        lgb_preds = train_group_kfold_lightgbm(
            cfg.model.fold,
            X,
            y,
            X_test,
            group,
            dict(cfg.params),
            cfg.model.verbose,
        )

        # Save test predictions
        test["pressure"] = lgb_preds
        submit_path = to_absolute_path(cfg.submit.path) + "/"
        test[["id", "pressure"]].to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
