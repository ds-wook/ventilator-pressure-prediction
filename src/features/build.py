import pandas as pd
from pandas import DataFrame


def add_features(df: DataFrame) -> DataFrame:
    """
    Loads Feature engineering Dataset
        Parameter:
            df: train or test dataset
        Return:
            df: feature engineering dataset
    """
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["u_in_cumsum"] = df.groupby(["breath_id"])["u_in"].cumsum()
    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()

    df["u_in_1st_derivative"] = (
        df["u_in"].diff().fillna(0) / df["time_step"].diff().fillna(0)
    ).fillna(0)
    df["expand_mean_1sr_der"] = (
        df.groupby("breath_id")["u_in_1st_derivative"]
        .expanding(2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_in_1st_der_mean10"] = (
        df.groupby("breath_id")["u_in_1st_derivative"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_in_lag4"] = (
        df.groupby("breath_id")["u_in"]
        .shift(4)
        .fillna(0)
        .reset_index(level=0, drop=True)
    )
    df["u_in_lag-4"] = (
        df.groupby("breath_id")["u_in"]
        .shift(-4)
        .fillna(0)
        .reset_index(level=0, drop=True)
    )

    df["time_diff"] = df["time_step"] - df["time_step"].shift(1).fillna(0)
    df["power"] = df["time_diff"] * df["u_in"]
    df["power_cumsum"] = df.groupby(["breath_id"])["power"].cumsum()

    df["u_in_gap"] = df["u_in"] - df["u_in"].shift(1).fillna(0)
    df["u_in_rate"] = df["u_in_gap"] / df["time_diff"]

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)

    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__min"] = df.groupby(["breath_id"])["u_in"].transform("min")

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["RC"] = df["R"] + df["C"]
    df = pd.get_dummies(df)

    df["last_value_u_in"] = df.groupby("breath_id")["u_in"].transform("last")
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]

    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )

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

    df["delta_u_in"] = abs(
        df.groupby(df["breath_id"])["u_in"].diff().fillna(0)
    ).reset_index(level=0, drop=True)
    df["delta_u_in_exp"] = (
        df.groupby(df["breath_id"])["delta_u_in"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["delta_rolling_10_mean"] = (
        df.groupby("breath_id")["delta_u_in"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["delta_rolling_10_max"] = (
        df.groupby("breath_id")["delta_u_in"]
        .rolling(window=10, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["work"] = (
        (df["u_in"] + df["u_in"].shift(1).fillna(0))
        / 2
        * df["time_step"].diff().fillna(0)
    ).clip(
        0,
    )
    df["work_roll_10"] = (
        df.groupby(df["breath_id"])["work"]
        .rolling(window=10, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["work_roll_15"] = (
        df.groupby(df["breath_id"])["work"]
        .rolling(window=15, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["u_in_rol_q0.1"] = (
        df.groupby(df["breath_id"])["u_in"]
        .rolling(window=10, min_periods=1)
        .quantile(0.1)
        .reset_index(level=0, drop=True)
    )
    df["u_in_rol_q0.25"] = (
        df.groupby(df["breath_id"])["u_in"]
        .rolling(window=10, min_periods=1)
        .quantile(0.25)
        .reset_index(level=0, drop=True)
    )
    df["u_in_rol_q0.5"] = (
        df.groupby(df["breath_id"])["u_in"]
        .rolling(window=10, min_periods=1)
        .quantile(0.5)
        .reset_index(level=0, drop=True)
    )
    df["u_in_rol_q0.75"] = (
        df.groupby(df["breath_id"])["u_in"]
        .rolling(window=10, min_periods=1)
        .quantile(0.75)
        .reset_index(level=0, drop=True)
    )
    df["u_in_rol_q0.9"] = (
        df.groupby(df["breath_id"])["u_in"]
        .rolling(window=10, min_periods=1)
        .quantile(0.9)
        .reset_index(level=0, drop=True)
    )

    df = df.fillna(0)
    return df
