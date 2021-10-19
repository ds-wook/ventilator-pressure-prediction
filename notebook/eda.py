# %%
import numpy as np
import pandas as pd

# %%
path = "../input/ventilator-pressure-prediction/"
train = pd.read_csv(path + "train.csv")
oof = np.load(path + "finetuning_lstm_oof.npy")
train["pressure"] = oof.flatten()
train[["id", "pressure"]].to_csv("finetuning_lstm_oof.csv", index=False)
# %%
oof["pressure"] = oof["oof"]
oof[["id", "pressure"]].to_csv(path + "ventilator-classification-train.csv", index=False)
# %%
