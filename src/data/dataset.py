import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["time_lag"] = df["time_step"].shift(2)
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
    df["ewm_u_in_mean"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(halflife=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["ewm_u_in_std"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(halflife=10)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["ewm_u_in_corr"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(halflife=10)
        .corr()
        .reset_index(level=0, drop=True)
    )

    df["rolling_10_mean"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_10_max"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=10, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )
    df["rolling_10_std"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=10, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["expand_mean"] = (
        df.groupby("breath_id")["u_in"]
        .expanding(2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["expand_max"] = (
        df.groupby("breath_id")["u_in"]
        .expanding(2)
        .max()
        .reset_index(level=0, drop=True)
    )
    df["expand_std"] = (
        df.groupby("breath_id")["u_in"]
        .expanding(2)
        .std()
        .reset_index(level=0, drop=True)
    )
    return df


def bilstm_data(df: pd.DataFrame, num: int) -> pd.DataFrame:
    df.rename(
        columns={f"pressure{i}": f"bilstm_pred{i}" for i in range(num)}, inplace=True
    )
    for i in range(num):
        df[f"bilstm_pred{i}_lag1"] = df.groupby("breath_id")[f"bilstm_pred{i}"].shift(1)
        df[f"bilstm_pred{i}_lag2"] = df.groupby("breath_id")[f"bilstm_pred{i}"].shift(2)
    df = df.fillna(0)

    return df
