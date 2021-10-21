import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from optimization.blend import get_best_weights


@hydra.main(config_path="../config/train/", config_name="blending.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + "train.csv")
    train_bilstm1 = pd.read_csv(path + "finetuning_train.csv")
    test_bilstm1 = pd.read_csv(path + "finetuning_test.csv")

    train_bilstm2 = pd.read_csv(path + "bilstm_train.csv")
    test_bilstm2 = pd.read_csv(path + "bilstm_test.csv")

    train_bilstm3 = pd.read_csv(path + "finetuning_lstm_oof.csv")
    test_bilstm3 = pd.read_csv(path + "finetuning_lstm_pred.csv")

    train_bilstm4 = pd.read_csv(path + "single_bilstm_train.csv")
    test_bilstm4 = pd.read_csv(path + "single_bilstm_test.csv")

    train_bilstm5 = pd.read_csv(path + "ventilator-classification-train.csv")
    test_bilstm5 = pd.read_csv(path + "ventilator-classification-test.csv")

    train_bilstm6 = pd.read_csv(path + "rescaling_bilstm_train.csv")
    test_bilstm6 = pd.read_csv(path + "rescaling_bilstm_test.csv")

    train_bilstm7 = pd.read_csv(path + "fine-tune-regression-train.csv")
    test_bilstm7 = pd.read_csv(path + "fine-tune-regression-test.csv")

    train_linear = pd.read_csv(path + "automl-train.csv")
    test_linear = pd.read_csv(path + "automl-test.csv")

    oofs = [
        train_bilstm1.pressure.values,
        train_bilstm2.pressure.values,
        train_bilstm3.pressure.values,
        train_bilstm4.pressure.values,
        train_bilstm5.pressure.values,
        train_bilstm6.pressure.values,
        train_bilstm7.pressure.values,
        train_linear.pressure.values,
    ]

    preds = [
        test_bilstm1.pressure.values,
        test_bilstm2.pressure.values,
        test_bilstm3.pressure.values,
        test_bilstm4.pressure.values,
        test_bilstm5.pressure.values,
        test_bilstm6.pressure.values,
        test_bilstm7.pressure.values,
        test_linear.pressure.values,
    ]

    mean_weight = get_best_weights(oofs, train.pressure.values)

    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")

    blending_preds = np.average(preds, weights=mean_weight, axis=0)
    submission["pressure"] = blending_preds

    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
