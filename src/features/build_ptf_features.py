from pathlib import Path
import numpy as np
import pandas as pd


PROCESSED_PATH = Path("data/processed/ptf/ptf_processed.parquet")
FEATURES_PATH = Path("data/features/ptf/ptf_features.parquet")

TARGET_HORIZON = 24

LAGS = [1, 3, 24, 48, 72, 168, 336]
ROLLING_WINDOWS = [24, 168]


def load_processed_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {path}")

    df = pd.read_parquet(path)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = ["date", "ptf", "priceusd", "priceeur"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ptf"] = pd.to_numeric(df["ptf"], errors="coerce")
    df["priceusd"] = pd.to_numeric(df["priceusd"], errors="coerce")
    df["priceeur"] = pd.to_numeric(df["priceeur"], errors="coerce")

    df = df.dropna(subset=["date", "ptf"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="first")
    df = df.set_index("date")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_currency_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["priceusd"] = df["priceusd"].replace(0, np.nan)
    df["priceeur"] = df["priceeur"].replace(0, np.nan)

    df["ptf_usd_ratio"] = df["ptf"] / df["priceusd"]
    df["ptf_eur_ratio"] = df["ptf"] / df["priceeur"]

    return df


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["ptf"].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()

    base = df["ptf"].shift(1)

    for window in windows:
        df[f"rolling_mean_{window}"] = base.rolling(window).mean()
        df[f"rolling_std_{window}"] = base.rolling(window).std()
        df[f"rolling_min_{window}"] = base.rolling(window).min()
        df[f"rolling_max_{window}"] = base.rolling(window).max()

    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["diff_1"] = df["ptf"].diff(1)
    df["diff_24"] = df["ptf"].diff(24)
    df["diff_168"] = df["ptf"].diff(168)

    return df


def add_clipped_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    q_low = df["ptf"].quantile(0.01)
    q_high = df["ptf"].quantile(0.99)

    df["ptf_clipped"] = df["ptf"].clip(lower=q_low, upper=q_high)

    df["lag_24_clipped"] = df["ptf_clipped"].shift(24)
    df["lag_168_clipped"] = df["ptf_clipped"].shift(168)

    base_clipped = df["ptf_clipped"].shift(1)

    df["rolling_mean_24_clipped"] = base_clipped.rolling(24).mean()
    df["rolling_mean_168_clipped"] = base_clipped.rolling(168).mean()

    df["rolling_std_24_clipped"] = base_clipped.rolling(24).std()
    df["rolling_std_168_clipped"] = base_clipped.rolling(168).std()

    return df


def add_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    q_low = df["ptf"].quantile(0.01)
    q_high = df["ptf"].quantile(0.99)

    df["spike_high"] = (df["ptf"].shift(1) > q_high).astype("int8")
    df["spike_low"] = (df["ptf"].shift(1) < q_low).astype("int8")

    ptf_diff_abs = df["ptf"].diff().abs()
    rolling_std_24 = df["ptf"].shift(1).rolling(24).std()
    rolling_std_168 = df["ptf"].shift(1).rolling(168).std()

    df["spike_jump_24"] = (ptf_diff_abs > (3 * rolling_std_24)).astype("int8")
    df["spike_jump_168"] = (ptf_diff_abs > (3 * rolling_std_168)).astype("int8")

    return df


def add_target(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df["ptf"].shift(-horizon)
    return df


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna()
    return df


def build_feature_list() -> list[str]:
    feature_cols = [
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",

        "ptf_usd_ratio",
        "ptf_eur_ratio",

        "lag_1",
        "lag_3",
        "lag_24",
        "lag_48",
        "lag_72",
        "lag_168",
        "lag_336",

        "rolling_mean_24",
        "rolling_std_24",
        "rolling_min_24",
        "rolling_max_24",
        "rolling_mean_168",
        "rolling_std_168",
        "rolling_min_168",
        "rolling_max_168",

        "diff_1",
        "diff_24",
        "diff_168",

        "lag_24_clipped",
        "lag_168_clipped",
        "rolling_mean_24_clipped",
        "rolling_mean_168_clipped",
        "rolling_std_24_clipped",
        "rolling_std_168_clipped",

        "spike_high",
        "spike_low",
        "spike_jump_24",
        "spike_jump_168",
    ]
    return feature_cols


def save_features(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        path,
        engine="pyarrow",
        compression="snappy",
        index=True,
    )


def main() -> None:
    print("Loading processed data...")
    df = load_processed_data(PROCESSED_PATH)

    print("Validating required columns...")
    validate_required_columns(df)

    print("Preparing base dataframe...")
    df = prepare_base_dataframe(df)

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding currency features...")
    df = add_currency_features(df)

    print("Adding lag features...")
    df = add_lag_features(df, lags=LAGS)

    print("Adding rolling features...")
    df = add_rolling_features(df, windows=ROLLING_WINDOWS)

    print("Adding diff features...")
    df = add_diff_features(df)

    print("Adding clipped robust features...")
    df = add_clipped_features(df)

    print("Adding spike features...")
    df = add_spike_features(df)

    print("Adding target...")
    df = add_target(df, horizon=TARGET_HORIZON)

    print("Dropping rows with NaN values...")
    df = finalize_features(df)

    feature_cols = build_feature_list()
    target_col = "target"

    missing_feature_cols = [col for col in feature_cols if col not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing engineered feature columns: {missing_feature_cols}")

    final_cols = feature_cols + [target_col]
    df_final = df[final_cols].copy()

    print("Saving feature dataset...")
    save_features(df_final, FEATURES_PATH)

    print("Done.")
    print(f"Final shape: {df_final.shape}")
    print(f"Saved to: {FEATURES_PATH}")
    print("Feature columns:")
    for col in feature_cols:
        print(f" - {col}")


if __name__ == "__main__":
    main()