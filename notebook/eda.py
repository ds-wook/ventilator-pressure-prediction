# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
path = "../input/ventilator-pressure-prediction/"
oof = pd.read_csv(path + "single_bilstm_train.csv")
train = pd.read_csv(path + "train.csv")
mean_absolute_error(train.pressure, oof.pressure)

# %%
