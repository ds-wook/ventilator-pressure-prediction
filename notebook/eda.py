# %%
import numpy as np
import pandas as pd

# %%
path = "../input/ventilator-pressure-prediction/"
oof = pd.read_csv(path + "transformer-train.csv")

oof[["id", "pressure"]].to_csv("transformer-train.csv", index=False)

# %%
