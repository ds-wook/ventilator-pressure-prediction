import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error

from optimization.blend import get_best_weights


@hydra.main(config_path="../config/train/", config_name="blending.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train = pd.read_csv(path + "train.csv")
    submission = pd.read_csv(path + "sample_submission.csv")
    train = train[train["u_out"] < 1].reset_index(drop=True)
    target = train["pressure"]
    train.drop("pressure", axis=1, inplace=True)

    lgbm_oofs = pd.read_csv(path + "five_model_lightgbm_oof.csv")
    lgbm_preds = pd.read_csv(submit_path + "five_model_lightgbm_preds.csv")

    train_bilstm1 = pd.read_csv(path + "finetuning_train.csv")
    train_bilstm1 = pd.merge(train_bilstm1, train, on="id")
    test_bilstm1 = pd.read_csv(path + "finetuning_test.csv")

    train_bilstm4 = pd.read_csv(path + "single_bilstm_train.csv")
    train_bilstm4 = pd.merge(train_bilstm4, train, on="id")
    test_bilstm4 = pd.read_csv(path + "single_bilstm_test.csv")

    train_linear = pd.read_csv(path + "automl-train.csv")
    train_linear = pd.merge(train_linear, train, on="id")
    test_linear = pd.read_csv(path + "automl-test.csv")

    oofs = [
        train_bilstm1.pressure.values,
        train_bilstm4.pressure.values,
        train_linear.pressure.values,
        lgbm_oofs.lgbm_preds.values,
    ]

    preds = [
        test_bilstm1.pressure.values,
        test_bilstm4.pressure.values,
        test_linear.pressure.values,
        lgbm_preds.pressure.values,
    ]

    best_weights = get_best_weights(oofs, target.values)

    oof_preds = np.average(oofs, weights=best_weights, axis=0)
    print(f"OOF Score: {mean_absolute_error(target, oof_preds)}")

    blending_preds = np.average(preds, weights=best_weights, axis=0)
    submission["pressure"] = blending_preds

    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
