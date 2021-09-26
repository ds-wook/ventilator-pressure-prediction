import pandas as pd


def load_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["last_value_u_in"] = df.groupby("breath_id")["u_in"].transform("last")
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
    df = df.fillna(0)

    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)

    # max value of u_in and u_out for each breath
    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_out__max"] = df.groupby(["breath_id"])["u_out"].transform("max")

    # difference between consequitive values
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]

    # from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
    df.loc[df["time_step"] == 0, "u_in_diff"] = 0
    df.loc[df["time_step"] == 0, "u_out_diff"] = 0

    # difference between the current value of u_in and the max value within the breath
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )

    # OHE
    df = df.merge(
        pd.get_dummies(df["R"], prefix="R"), left_index=True, right_index=True
    ).drop(["R"], axis=1)
    df = df.merge(
        pd.get_dummies(df["C"], prefix="C"), left_index=True, right_index=True
    ).drop(["C"], axis=1)
    df = df.merge(
        pd.get_dummies(df["R__C"], prefix="R__C"), left_index=True, right_index=True
    ).drop(["R__C"], axis=1)

    df["u_in_cumsum"] = df.groupby(["breath_id"])["u_in"].cumsum()
    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()

    return df
