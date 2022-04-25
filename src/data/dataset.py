from pathlib import Path
from typing import Tuple

import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pandas import DataFrame, Series

from features.build import add_features
from utils.utils import reduce_mem_usage


def load_train_dataset(config: DictConfig) -> Tuple[DataFrame, Series]:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        (train dataset, target dataset)
    """
    path = Path(get_original_cwd()) / config.dataset.path
    train = pd.read_csv(path / config.dataset.train)

    train_bilstm = pd.read_csv(path + "bilstm_train.csv")
    train["bilstm_pred"] = train_bilstm["pressure"]
    del train_bilstm

    train_bilstm = pd.read_csv(path + "finetuning_lstm_oof.csv")
    train["finetuning_lstm_pred"] = train_bilstm["pressure"]
    del train_bilstm

    train_bilstm = pd.read_csv(path + "gb-vpp-another-lstm-train.csv")
    train["another_lstm_pred"] = train_bilstm["pressure"]
    del train_bilstm

    train_bilstm = pd.read_csv(path + "ventilator-classification-train.csv")
    train["ventilator_classification_pred"] = train_bilstm["pressure"]
    del train_bilstm
    train_bilstm = pd.read_csv(path + "rescaling_bilstm_train.csv")
    train["rescaling_bilstm_pred"] = train_bilstm["pressure"]
    del train_bilstm

    train_bilstm = pd.read_csv(path + "gb-vpp-median-lstm-train.csv")
    train["median_pred"] = train_bilstm["pressure"]
    del train_bilstm

    train_bilstm = pd.read_csv(path + "dnn_lstm_train.csv")
    train["dnn_pred"] = train_bilstm["pressure"]
    del train_bilstm

    train_resnet = pd.read_csv(path + "lstm_resnet_train.csv")
    train["resnet_pred"] = train_resnet["pressure"]
    del train_resnet

    train = add_features(train)
    train = reduce_mem_usage(train)

    train_x = train[config.dataset.features]
    train_y = train[config.dataset.target]

    return train_x, train_y


def load_test_dataset(config: DictConfig) -> DataFrame:
    """
    Load test dataset
    Args:
        config: config
    Returns:
        test dataset
    """
    path = Path(get_original_cwd()) / config.dataset.path
    test = pd.read_csv(path / config.dataset.test)
    test_bilstm = pd.read_csv(path + "bilstm_test.csv")

    test["bilstm_pred"] = test_bilstm["pressure"]
    del test_bilstm

    test_bilstm = pd.read_csv(path + "finetuning_lstm_pred.csv")

    test["finetuning_lstm_pred"] = test_bilstm["pressure"]
    del test_bilstm

    test_bilstm = pd.read_csv(path + "gb-vpp-another-lstm-preds.csv")

    test["another_lstm_pred"] = test_bilstm["pressure"]
    del test_bilstm

    test_bilstm = pd.read_csv(path + "ventilator-classification-test.csv")
    test["ventilator_classification_pred"] = test_bilstm["pressure"]
    del test_bilstm

    test_bilstm = pd.read_csv(path + "rescaling_bilstm_test.csv")
    test["rescaling_bilstm_pred"] = test_bilstm["pressure"]
    del test_bilstm

    test_bilstm = pd.read_csv(path + "gb-vpp-median-lstm-preds.csv")
    test["median_pred"] = test_bilstm["pressure"]
    del test_bilstm

    test_bilstm = pd.read_csv(path + "dnn_lstm_preds.csv")
    test["dnn_pred"] = test_bilstm["pressure"]
    del test_bilstm

    test_resnet = pd.read_csv(path + "lstm_resnet_test.csv")
    test["resnet_pred"] = test_resnet["pressure"]
    del test_resnet

    test = add_features(test)
    test = reduce_mem_usage(test)

    test_x = test[config.dataset.features]

    return test_x
