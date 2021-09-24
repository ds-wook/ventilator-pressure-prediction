from typing import List, Tuple

import pandas as pd

from utils.utils import reduce_mem_usage


def aggregation(
    input_df: pd.DataFrame,
    group_key: str,
    group_values: List[str],
    agg_methods: List[str],
) -> pd.DataFrame:
    """ref:https://github.com/pfnet-research/xfeat/blob/master/xfeat/helper.py"""
    new_df = []
    for agg_method in agg_methods:
        for col in group_values:
            if callable(agg_method):
                agg_method_name = agg_method.__name__
            else:
                agg_method_name = agg_method
            new_col = f"agg_{agg_method_name}_{col}_grpby_{group_key}"
            df_agg = (
                input_df[[col] + [group_key]].groupby(group_key)[[col]].agg(agg_method)
            )
            df_agg.columns = [new_col]
            new_df.append(df_agg)

    _df = pd.concat(new_df, axis=1).reset_index()
    output_df = pd.merge(input_df[[group_key]], _df, on=group_key, how="left")
    return output_df.drop(group_key, axis=1)


def get_raw_features(input_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["R", "C", "time_step", "u_in", "u_out"]
    output_df = input_df[cols].copy()
    return output_df


def get_cross_features(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = pd.DataFrame()
    output_df["R+C"] = (input_df["R"].astype(str) + input_df["C"].astype(str)).astype(
        int
    )
    return output_df


def get_shift_grpby_breath_id_features(input_df: pd.DataFrame) -> pd.DataFrame:
    # future feats
    shift_times = [-1, -2, 1, 2]
    group_key = "breath_id"

    group_values = ["u_in"]

    output_df = pd.DataFrame()
    for t in shift_times:
        _df = input_df.groupby(group_key)[group_values].shift(t)
        _df.columns = [f"shift={t}_{col}_grpby_{group_key}" for col in group_values]
        output_df = pd.concat([output_df, _df], axis=1)
    return output_df


def get_diff_grpby_breath_id_features(input_df: pd.DataFrame) -> pd.DataFrame:
    # future feats
    diff_times = [-1, -2, 1, 2]
    group_key = "breath_id"
    group_values = ["u_in"]

    output_df = pd.DataFrame()
    for t in diff_times:
        _df = input_df.groupby(group_key)[group_values].shift(t)
        _df.columns = [f"diff={t}_{col}_grpby_{group_key}" for col in group_values]
        output_df = pd.concat([output_df, _df], axis=1)
    return output_df


def get_cumsum_grpby_breath_id_features(input_df: pd.DataFrame) -> pd.DataFrame:
    group_key = "breath_id"
    group_values = ["time_step", "u_in", "u_out"]

    output_df = pd.DataFrame()
    for group_val in group_values:
        col_name = f"agg_cumsum_{group_val}_grpby_{group_key}"
        output_df[col_name] = input_df.groupby(group_key)[group_val].cumsum()

    return output_df


def get_time_step_cat_features(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = pd.DataFrame()
    output_df["time_step_cat"] = input_df["time_step"].copy()
    output_df.loc[input_df["time_step"] < 1, "time_step_cat"] = 0
    output_df.loc[
        (1 < input_df["time_step"]) & (input_df["time_step"] < 1.5), "time_step_cat"
    ] = 1
    output_df.loc[1.5 < input_df["time_step"], "time_step_cat"] = 2
    return output_df


def get_breath_id_pivot_features(input_df: pd.DataFrame) -> pd.DataFrame:
    _df = input_df[["breath_id", "time_step", "u_in"]].copy()
    _df["time_step_id"] = _df.groupby(["breath_id"])["time_step"].rank(ascending=True)
    _df = pd.pivot_table(_df, columns="time_step_id", index="breath_id", values="u_in")
    _df.columns = [f"time_step_id={int(i):02}_u_in" for i in _df.columns]
    output_df = pd.merge(
        input_df[["breath_id"]], _df, left_on="breath_id", right_index=True, how="left"
    )
    return output_df.drop("breath_id", axis=1)


def get_agg_breath_id_whole_features(whole_df: pd.DataFrame) -> pd.DataFrame:
    # do not have to use whole_df
    group_key = "breath_id"
    group_values = ["u_in"]
    agg_methods = ["mean", "std", "median", "max", "sum"]

    output_df = aggregation(whole_df, group_key, group_values, agg_methods)

    # z-score
    z_col_name = _get_agg_col_name(group_key, group_values, ["z-score"])
    m_col_name = _get_agg_col_name(group_key, group_values, ["mean"])
    # s_col_name = _get_agg_col_name(group_key, group_values, ["std"])

    output_df[z_col_name] = (
        whole_df[group_values].values - output_df[m_col_name].values
    ) / (output_df[m_col_name].values + 1e-8)
    return output_df


def _get_agg_col_name(group_key: str, group_values: List[str], agg_methods):
    out_cols = [
        f"agg_{agg_method}_{group_val}_grpby_{group_key}"
        for group_val in group_values
        for agg_method in agg_methods
    ]

    return out_cols


def get_features(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = pd.DataFrame()
    funcs = [
        get_raw_features,
        get_cross_features,
        get_shift_grpby_breath_id_features,
        get_diff_grpby_breath_id_features,
        get_time_step_cat_features,
        get_breath_id_pivot_features,
        get_cumsum_grpby_breath_id_features,
    ]
    for func in funcs:
        print(func.__name__)
        _df = func(input_df)
        _df = reduce_mem_usage(_df)
        output_df = pd.concat([output_df, _df], axis=1)

    return output_df


def get_whole_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    whole_df = pd.concat([train, test]).reset_index(drop=True)
    output_df = pd.DataFrame()
    funcs = [
        get_agg_breath_id_whole_features,
    ]

    if not funcs:
        return pd.DataFrame(), pd.DataFrame()

    for func in funcs:
        print(func.__name__)
        _df = func(whole_df)
        _df = reduce_mem_usage(_df)
        output_df = pd.concat([output_df, _df], axis=1)

    train_x = output_df.iloc[: len(train)]
    test_x = output_df.iloc[len(train) :].reset_index(drop=True)

    return train_x, test_x


def preprocess(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    # whole feature
    train_x, test_x = get_whole_features(train, test)
    train_x = pd.concat([train_x, get_features(train)], axis=1)
    test_x = pd.concat([test_x, get_features(test)], axis=1)
    train_y = train["pressure"]
    return train_x, train_y, test_x


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")
    train["last_value_u_in"] = train.groupby("breath_id")["u_in"].transform("last")
    train["u_in_lag1"] = train.groupby("breath_id")["u_in"].shift(1)
    train["u_out_lag1"] = train.groupby("breath_id")["u_out"].shift(1)
    train["u_in_lag_back1"] = train.groupby("breath_id")["u_in"].shift(-1)
    train["u_out_lag_back1"] = train.groupby("breath_id")["u_out"].shift(-1)
    train["u_in_lag2"] = train.groupby("breath_id")["u_in"].shift(2)
    train["u_out_lag2"] = train.groupby("breath_id")["u_out"].shift(2)
    train["u_in_lag_back2"] = train.groupby("breath_id")["u_in"].shift(-2)
    train["u_out_lag_back2"] = train.groupby("breath_id")["u_out"].shift(-2)
    train = train.fillna(0)

    train["R__C"] = train["R"].astype(str) + "__" + train["C"].astype(str)

    # max value of u_in and u_out for each breath
    train["breath_id__u_in__max"] = train.groupby(["breath_id"])["u_in"].transform(
        "max"
    )
    train["breath_id__u_out__max"] = train.groupby(["breath_id"])["u_out"].transform(
        "max"
    )

    # difference between consequitive values
    train["u_in_diff1"] = train["u_in"] - train["u_in_lag1"]
    train["u_out_diff1"] = train["u_out"] - train["u_out_lag1"]
    train["u_in_diff2"] = train["u_in"] - train["u_in_lag2"]
    train["u_out_diff2"] = train["u_out"] - train["u_out_lag2"]
    # from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
    train.loc[train["time_step"] == 0, "u_in_diff"] = 0
    train.loc[train["time_step"] == 0, "u_out_diff"] = 0

    # difference between the current value of u_in and the max value within the breath
    train["breath_id__u_in__diffmax"] = (
        train.groupby(["breath_id"])["u_in"].transform("max") - train["u_in"]
    )
    train["breath_id__u_in__diffmean"] = (
        train.groupby(["breath_id"])["u_in"].transform("mean") - train["u_in"]
    )

    # OHE
    train = train.merge(
        pd.get_dummies(train["R"], prefix="R"), left_index=True, right_index=True
    ).drop(["R"], axis=1)
    train = train.merge(
        pd.get_dummies(train["C"], prefix="C"), left_index=True, right_index=True
    ).drop(["C"], axis=1)
    train = train.merge(
        pd.get_dummies(train["R__C"], prefix="R__C"), left_index=True, right_index=True
    ).drop(["R__C"], axis=1)

    train["u_in_cumsum"] = train.groupby(["breath_id"])["u_in"].cumsum()
    train["time_step_cumsum"] = train.groupby(["breath_id"])["time_step"].cumsum()

    # all the same for the test data
    test["last_value_u_in"] = test.groupby("breath_id")["u_in"].transform("last")
    test["u_in_lag1"] = test.groupby("breath_id")["u_in"].shift(1)
    test["u_out_lag1"] = test.groupby("breath_id")["u_out"].shift(1)
    test["u_in_lag_back1"] = test.groupby("breath_id")["u_in"].shift(-1)
    test["u_out_lag_back1"] = test.groupby("breath_id")["u_out"].shift(-1)
    test["u_in_lag2"] = test.groupby("breath_id")["u_in"].shift(2)
    test["u_out_lag2"] = test.groupby("breath_id")["u_out"].shift(2)
    test["u_in_lag_back2"] = test.groupby("breath_id")["u_in"].shift(-2)
    test["u_out_lag_back2"] = test.groupby("breath_id")["u_out"].shift(-2)
    test = test.fillna(0)
    test["R__C"] = test["R"].astype(str) + "__" + test["C"].astype(str)

    test["breath_id__u_in__max"] = test.groupby(["breath_id"])["u_in"].transform("max")
    test["breath_id__u_out__max"] = test.groupby(["breath_id"])["u_out"].transform(
        "max"
    )

    test["u_in_diff1"] = test["u_in"] - test["u_in_lag1"]
    test["u_out_diff1"] = test["u_out"] - test["u_out_lag1"]
    test["u_in_diff2"] = test["u_in"] - test["u_in_lag2"]
    test["u_out_diff2"] = test["u_out"] - test["u_out_lag2"]
    test.loc[test["time_step"] == 0, "u_in_diff"] = 0
    test.loc[test["time_step"] == 0, "u_out_diff"] = 0

    test["breath_id__u_in__diffmax"] = (
        test.groupby(["breath_id"])["u_in"].transform("max") - test["u_in"]
    )
    test["breath_id__u_in__diffmean"] = (
        test.groupby(["breath_id"])["u_in"].transform("mean") - test["u_in"]
    )

    test = test.merge(
        pd.get_dummies(test["R"], prefix="R"), left_index=True, right_index=True
    ).drop(["R"], axis=1)
    test = test.merge(
        pd.get_dummies(test["C"], prefix="C"), left_index=True, right_index=True
    ).drop(["C"], axis=1)
    test = test.merge(
        pd.get_dummies(test["R__C"], prefix="R__C"), left_index=True, right_index=True
    ).drop(["R__C"], axis=1)

    test["u_in_cumsum"] = test.groupby(["breath_id"])["u_in"].cumsum()
    test["time_step_cumsum"] = test.groupby(["breath_id"])["time_step"].cumsum()

    return train, test
