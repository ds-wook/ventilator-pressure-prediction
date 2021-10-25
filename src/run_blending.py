import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error

from optimization.blend import get_best_weights
from utils.utils import find_nearest


@hydra.main(config_path="../config/train/", config_name="blending.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train = pd.read_csv(path + "train.csv")
    all_pressure = np.sort(train.pressure.unique())
    submission = pd.read_csv(path + "sample_submission.csv")
    train = train[train["u_out"] < 1].reset_index(drop=True)
    target = train["pressure"]
    train.drop("pressure", axis=1, inplace=True)

    lgbm_oofs = pd.read_csv(path + "five_lstm_lightgbm_oof.csv")
    lgbm_preds = pd.read_csv(submit_path + "five_lstm_lightgbm_preds.csv")

    train_bilstm = pd.read_csv(path + "gb-vpp-another-lstm-train.csv")
    test_bilstm = pd.read_csv(path + "gb-vpp-another-lstm-preds.csv")

    oofs = [
        lgbm_oofs.lgbm_preds.values,
        train_bilstm.pressure.values,
    ]

    preds = [
        lgbm_preds.pressure.values,
        test_bilstm.pressure.values
    ]

    best_weights = get_best_weights(oofs, target.values)

    oof_preds = np.average(oofs, weights=best_weights, axis=0)
    print(f"OOF Score: {mean_absolute_error(target, oof_preds)}")

    blending_preds = np.average(preds, weights=best_weights, axis=0)

    submission["pressure"] = blending_preds
    submission["pressure"] = submission["pressure"].map(
        lambda x: find_nearest(all_pressure, x)
    )

    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
