import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    df["delta_t"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

    df["delta_cx"] = df["cx"].diff().fillna(0)
    df["delta_cy"] = df["cy"].diff().fillna(0)

    df["delta_vx"] = df["vx"].diff().fillna(0)
    df["delta_vy"] = df["vy"].diff().fillna(0)

    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    df["accel"] = np.sqrt(df["delta_vx"]**2 + df["delta_vy"]**2) / (df["delta_t"] + 1e-6)

    df["direction"] = np.arctan2(df["vy"], df["vx"])

    df["delta_area"] = df["area"].diff().fillna(0)
    df["rel_area_change"] = (df["delta_area"] / (df["area"].shift(1).replace(0, np.nan))).fillna(0)

    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5):
    df = df.copy()

    for col in ["vx", "vy", "speed", "accel", "area"]:
        df[f"{col}_roll_mean"] = df[col].rolling(window=window, min_periods=1).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)

    return df


def engineer_features(df: pd.DataFrame):
    if df.empty:
        print("[WARN] Empty dataframe passed to engineer_features()")
        return df

    df = add_temporal_features(df)
    df = add_rolling_features(df)
    return df