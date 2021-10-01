# %%
import numpy as np
import pandas as pd

# %%
path = "../input/ventilator-pressure-prediction/"
train_bilstm = pd.read_csv(path + "lstm_train.csv")
test_bilstm = pd.read_csv(path + "lstm_test.csv")
train_bilstm.drop("Unnamed: 0", axis=1, inplace=True)
test_bilstm.drop("Unnamed: 0", axis=1, inplace=True)
train_bilstm.to_csv(path + "lstm_train.csv", index=False)
test_bilstm.to_csv(path + "lstm_test.csv", index=False)
# %%
