from typing import Tuple

import pandas as pd


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")
    train["last_value_u_in"] = train.groupby("breath_id")["u_in"].transform("last")
    train["u_in_lag"] = train["u_in"].shift(1)
    train["u_out_lag"] = train["u_out"].shift(1)
    train = train.fillna(0)

    train["u_in_diff"] = train["u_in"] - train["u_in_lag"]
    train["u_out_diff"] = train["u_out"] - train["u_out_lag"]

    train["breath_id__u_in__diffmax"] = (
        train.groupby(["breath_id"])["u_in"].transform("max") - train["u_in"]
    )
    train["breath_id__u_in__diffmean"] = (
        train.groupby(["breath_id"])["u_in"].transform("mean") - train["u_in"]
    )

    test["last_value_u_in"] = test.groupby("breath_id")["u_in"].transform("last")
    test["u_in_lag"] = test["u_in"].shift(1)
    test["u_out_lag"] = test["u_out"].shift(1)
    test = test.fillna(0)

    test["u_in_diff"] = test["u_in"] - test["u_in_lag"]
    test["u_out_diff"] = test["u_out"] - test["u_out_lag"]

    test["breath_id__u_in__diffmax"] = (
        test.groupby(["breath_id"])["u_in"].transform("max") - test["u_in"]
    )
    test["breath_id__u_in__diffmean"] = (
        test.groupby(["breath_id"])["u_in"].transform("mean") - test["u_in"]
    )

    return train, test
