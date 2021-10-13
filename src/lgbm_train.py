import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error

from data.dataset import add_features, bilstm_data
from trainer.boosting_tree import LGBMTrainer
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

    columns = [
        col
        for col in train.columns
        if col not in ["id", "breath_id", "pressure"]
    ]

    train_x = train[columns]
    train_y = train["pressure"]
    test_x = test[columns]
    groups = train["breath_id"]

    lgbm_trainer = LGBMTrainer(cfg.model.fold, mean_absolute_error)
    lgbm_trainer.train(train_x, train_y, groups, dict(cfg.params), cfg.model.verbose)
    lgbm_preds = lgbm_trainer.predict(test_x)
    lgbm_preds = lgbm_trainer.postprocess(train, lgbm_preds)

    # Save test predictions
    submission["pressure"] = lgbm_preds

    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
