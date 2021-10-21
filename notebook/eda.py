# %%
import numpy as np
import pandas as pd

path = "../input/ventilator-pressure-prediction/"
oof = pd.read_csv(path + "fine-tune-regression-train.csv")
oof["pressure"] = oof["oof"]
oof[["id", "pressure"]].to_csv("fine-tune-regression-train.csv", index=False)

# %%
