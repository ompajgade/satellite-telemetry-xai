import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "mean", "std", "var", "skew", "kurtosis", "n_peaks",
    "duration", "len", "len_weighted", "gaps_squared",
    "var_div_duration", "var_div_len",
    "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks",
    "diff_var", "diff2_var"
]

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def split_train_test(df):
    train_df = df[df["train"] == 1]
    test_df = df[df["train"] == 0]
    return train_df, test_df

def prepare_features(df, scaler=None, fit=False):
    X = df[FEATURE_COLUMNS].values
    y = df["anomaly"].values

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler