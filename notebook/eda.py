# %%
import numpy as np
import pandas as pd

# %%
train = pd.read_csv("../input/ventilator-pressure-prediction/" + "train.csv")
# %%
path = "../submit/"

lstm_oof = np.load(path + "median_lstm_oof.npy")

# %%
train.head()
# %%
train["lstm_pred"] = lstm_oof
# %%
train.head()
# %%
from sklearn.metrics import mean_absolute_error

mean_absolute_error(train["pressure"], train["lstm_pred"])
# %%
train_bilstm = pd.read_csv(
    "../input/ventilator-pressure-prediction/" + "lstm_train.csv"
)
train_bilstm
# %%
train_bilstm["pressure10"] = lstm_oof
train_bilstm.to_csv(
    "../input/ventilator-pressure-prediction/finetuning_train.csv", index=False
)
# %%
test_bilstm = pd.read_csv("../input/ventilator-pressure-prediction/" + "lstm_test.csv")
fine_tuning = pd.read_csv(path + "median_lstm_preds.csv")
test_bilstm["pressure10"] = fine_tuning["pressure"]
test_bilstm.to_csv(
    "../input/ventilator-pressure-prediction/finetuning_test.csv", index=False
)
# %%
