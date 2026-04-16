from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_PTF_PATH = Path("data/processed/ptf/ptf_processed.parquet")
PROCESSED_WEATHER_PATH = Path("data/processed/weather/weather_processed.parquet")
PROCESSED_GENERATION_PATH = Path("data/processed/generation/generation_processed.parquet")
PROCESSED_CONSUMPTION_PATH = Path("data/processed/consumption/consumption_processed.parquet")
PROCESSED_SMF_PATH = Path("data/processed/smf/smf_processed.parquet")

TRAIN_FEATURES_PATH = Path("data/features/ptf/ptf_features.parquet")
INFERENCE_LATEST_PATH = Path("data/features/ptf/ptf_features_inference_latest.parquet")
INFERENCE_BACKFILL_PATH = Path("data/features/ptf/ptf_features_inference_backfill.parquet")

CLIP_PARAMS_PATH = Path("artifacts/feature_params/ptf_clip_params.json")

TARGET_HORIZON = 24

LAGS = [1, 3, 24, 48, 72, 168, 336]
ROLLING_WINDOWS = [24, 168]
WEATHER_LAGS = [1, 24, 168]
EXTERNAL_LAGS = [1, 24, 168]
EXTERNAL_ROLLING_WINDOWS = [24, 168]

DATE_COL = "date"
TARGET_COL = "target"

EPSILON = 1e-6

JUMP_FLAG_COLS = [
    "smf_smf",
    "gen_naturalGas",
    "gen_wind",
    "gen_sun",
]

def load_processed_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {path}")
    return pd.read_parquet(path)


def validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = [DATE_COL, "ptf", "priceusd", "priceeur"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def ensure_naive_datetime(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series, errors="coerce")
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_localize(None)
    return series


def resolve_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["_ptf_positive_priority"] = (df["ptf"] > 0).astype("int8")
    df["_ptf_sort_value"] = df["ptf"].fillna(float("-inf"))

    df = df.sort_values(
        by=[DATE_COL, "_ptf_positive_priority", "_ptf_sort_value"],
        ascending=[True, False, False],
        kind="mergesort",
    )

    df = df.drop_duplicates(subset=[DATE_COL], keep="first").copy()
    df = df.drop(columns=["_ptf_positive_priority", "_ptf_sort_value"])

    return df


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[DATE_COL] = ensure_naive_datetime(df[DATE_COL])

    df["ptf"] = pd.to_numeric(df["ptf"], errors="coerce")
    df["priceusd"] = pd.to_numeric(df["priceusd"], errors="coerce")
    df["priceeur"] = pd.to_numeric(df["priceeur"], errors="coerce")

    df = df.dropna(subset=[DATE_COL, "ptf"]).copy()
    df = df.sort_values(DATE_COL, kind="mergesort").copy()

    df = resolve_duplicate_timestamps(df)

    df = df.set_index(DATE_COL)
    df = df.sort_index()

    return df


def load_weather_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed weather parquet not found: {path}")

    df = pd.read_parquet(path).copy()

    if DATE_COL not in df.columns:
        raise ValueError(f"Weather dataset must include '{DATE_COL}' column.")

    df[DATE_COL] = ensure_naive_datetime(df[DATE_COL])

    df = df.dropna(subset=[DATE_COL]).copy()
    df = df.sort_values(DATE_COL, kind="mergesort")
    df = df.drop_duplicates(subset=[DATE_COL], keep="last").copy()

    helper_drop_cols = [
        col
        for col in df.columns
        if col != DATE_COL
        and (
            col.startswith("_chunk")
            or col.startswith("_source")
            or col.startswith("_meta")
            or col.lower().startswith("unnamed")
        )
    ]
    if helper_drop_cols:
        df = df.drop(columns=helper_drop_cols)

    weather_cols = [col for col in df.columns if col != DATE_COL]
    rename_map = {}

    for col in weather_cols:
        if not col.startswith("weather_"):
            rename_map[col] = f"weather_{col}"

    if rename_map:
        df = df.rename(columns=rename_map)

    calendar_tokens = [
        "year",
        "month",
        "day",
        "hour",
        "weekday",
        "week",
        "quarter",
    ]

    bad_weather_cols = []
    for col in df.columns:
        if col == DATE_COL:
            continue

        col_lower = col.lower()

        if any(token in col_lower for token in calendar_tokens):
            bad_weather_cols.append(col)
            continue

        if df[col].nunique(dropna=True) <= 1:
            bad_weather_cols.append(col)

    if bad_weather_cols:
        df = df.drop(columns=sorted(set(bad_weather_cols)))

    return df


def build_generation_datetime(df: pd.DataFrame) -> pd.Series:
    date_series = ensure_naive_datetime(df["date"])

    has_hour_info = date_series.notna() & (
        (date_series.dt.hour != 0)
        | (date_series.dt.minute != 0)
        | (date_series.dt.second != 0)
    )
    if has_hour_info.any():
        return date_series

    if "hour" not in df.columns:
        return date_series

    hour_num = pd.to_numeric(df["hour"], errors="coerce")
    valid_hour = hour_num.dropna()

    if valid_hour.empty:
        return date_series

    if valid_hour.min() >= 1 and valid_hour.max() <= 24:
        hour_offset = hour_num - 1
    else:
        hour_offset = hour_num

    return date_series + pd.to_timedelta(hour_offset.fillna(0), unit="h")


def build_consumption_datetime(df: pd.DataFrame) -> pd.Series:
    date_series = ensure_naive_datetime(df["date"])

    has_hour_info = date_series.notna() & (
        (date_series.dt.hour != 0)
        | (date_series.dt.minute != 0)
        | (date_series.dt.second != 0)
    )
    if has_hour_info.any():
        return date_series

    if "time" not in df.columns:
        return date_series

    time_str = df["time"].astype(str).str.strip()
    combined = pd.to_datetime(
        date_series.dt.strftime("%Y-%m-%d") + " " + time_str,
        errors="coerce",
    )

    return combined.fillna(date_series)


def load_generation_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed generation parquet not found: {path}")

    df = pd.read_parquet(path).copy()

    if "date" not in df.columns:
        raise ValueError("Generation dataset must include 'date' column.")

    df[DATE_COL] = build_generation_datetime(df)
    df = df.dropna(subset=[DATE_COL]).copy()
    df = df.sort_values(DATE_COL, kind="mergesort")

    helper_drop_cols = [
        col
        for col in df.columns
        if col.lower() in {"hour", "year", "month", "_chunk_start", "_chunk_end"}
        or col.startswith("_chunk")
        or col.startswith("_source")
        or col.startswith("_meta")
        or col.lower().startswith("unnamed")
    ]
    if helper_drop_cols:
        df = df.drop(columns=helper_drop_cols, errors="ignore")

    rename_map = {}
    for col in df.columns:
        if col == DATE_COL:
            continue
        rename_map[col] = f"gen_{col}"

    df = df.rename(columns=rename_map)

    numeric_cols = [col for col in df.columns if col != DATE_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=[DATE_COL], keep="last").copy()

    return df


def load_consumption_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed consumption parquet not found: {path}")

    df = pd.read_parquet(path).copy()

    if "date" not in df.columns:
        raise ValueError("Consumption dataset must include 'date' column.")

    df[DATE_COL] = build_consumption_datetime(df)
    df = df.dropna(subset=[DATE_COL]).copy()
    df = df.sort_values(DATE_COL, kind="mergesort")

    helper_drop_cols = [
        col
        for col in df.columns
        if col.lower() in {"time", "year", "month", "_chunk_start", "_chunk_end"}
        or col.startswith("_chunk")
        or col.startswith("_source")
        or col.startswith("_meta")
        or col.lower().startswith("unnamed")
    ]
    if helper_drop_cols:
        df = df.drop(columns=helper_drop_cols, errors="ignore")

    rename_map = {}
    for col in df.columns:
        if col == DATE_COL:
            continue
        rename_map[col] = f"cons_{col}"

    df = df.rename(columns=rename_map)

    numeric_cols = [col for col in df.columns if col != DATE_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=[DATE_COL], keep="last").copy()

    return df


def load_smf_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed smf parquet not found: {path}")

    df = pd.read_parquet(path).copy()

    if "date" not in df.columns:
        raise ValueError("SMF dataset must include 'date' column.")

    df[DATE_COL] = ensure_naive_datetime(df["date"])
    df = df.dropna(subset=[DATE_COL]).copy()
    df = df.sort_values(DATE_COL, kind="mergesort")

    helper_drop_cols = [
        col
        for col in df.columns
        if col.lower() in {"hour", "year", "month", "_chunk_start", "_chunk_end"}
        or col.startswith("_chunk")
        or col.startswith("_source")
        or col.startswith("_meta")
        or col.lower().startswith("unnamed")
    ]
    if helper_drop_cols:
        df = df.drop(columns=helper_drop_cols, errors="ignore")

    rename_map = {}
    for col in df.columns:
        if col == DATE_COL:
            continue
        rename_map[col] = f"smf_{col}"

    df = df.rename(columns=rename_map)

    numeric_cols = [col for col in df.columns if col != DATE_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=[DATE_COL], keep="last").copy()

    return df


def merge_external_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    generation_df: pd.DataFrame | None = None,
    consumption_df: pd.DataFrame | None = None,
    smf_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy().reset_index()

    if weather_df is not None:
        df = df.merge(weather_df, on=DATE_COL, how="left")

    if generation_df is not None:
        df = df.merge(generation_df, on=DATE_COL, how="left")

    if consumption_df is not None:
        df = df.merge(consumption_df, on=DATE_COL, how="left")

    if smf_df is not None:
        df = df.merge(smf_df, on=DATE_COL, how="left")

    df = df.sort_values(DATE_COL, kind="mergesort").set_index(DATE_COL)
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

    base = df["ptf"].shift(1)
    df["diff_1"] = base.diff(1)
    df["diff_24"] = base.diff(24)
    df["diff_168"] = base.diff(168)

    return df


def compute_clip_bounds_from_train(df: pd.DataFrame) -> tuple[float, float]:
    q_low = float(df["ptf"].quantile(0.01))
    q_high = float(df["ptf"].quantile(0.99))
    return q_low, q_high


def save_clip_params(q_low: float, q_high: float, path: str | Path = CLIP_PARAMS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "q_low": float(q_low),
        "q_high": float(q_high),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_clip_params(path: str | Path = CLIP_PARAMS_PATH) -> tuple[float, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clip params file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "q_low" not in payload or "q_high" not in payload:
        raise KeyError(f"'q_low' and 'q_high' must exist in {path}")

    return float(payload["q_low"]), float(payload["q_high"])


def add_clipped_features(df: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    df = df.copy()

    df["ptf_clipped"] = df["ptf"].clip(lower=q_low, upper=q_high)

    df["lag_24_clipped"] = df["ptf_clipped"].shift(24)
    df["lag_168_clipped"] = df["ptf_clipped"].shift(168)

    base_clipped = df["ptf_clipped"].shift(1)

    df["rolling_mean_24_clipped"] = base_clipped.rolling(24).mean()
    df["rolling_mean_168_clipped"] = base_clipped.rolling(168).mean()
    df["rolling_std_24_clipped"] = base_clipped.rolling(24).std()
    df["rolling_std_168_clipped"] = base_clipped.rolling(168).std()

    return df


def add_spike_features(df: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    df = df.copy()

    prev_ptf = df["ptf"].shift(1)

    df["spike_high"] = (prev_ptf > q_high).astype("int8")
    df["spike_low"] = (prev_ptf < q_low).astype("int8")

    ptf_diff_abs = prev_ptf.diff().abs()
    rolling_std_24 = prev_ptf.rolling(24).std()
    rolling_std_168 = prev_ptf.rolling(168).std()

    df["spike_jump_24"] = (ptf_diff_abs > (3 * rolling_std_24)).astype("int8")
    df["spike_jump_168"] = (ptf_diff_abs > (3 * rolling_std_168)).astype("int8")

    return df

def get_weather_feature_columns(df: pd.DataFrame) -> list[str]:
    base_weather_cols = []

    engineered_tokens = [
        "_lag_",
        "_rolling_mean_",
        "_rolling_std_",
        "_rolling_min_",
        "_rolling_max_",
    ]

    banned_tokens = [
        "year",
        "month",
        "day",
        "hour",
        "weekday",
        "week",
        "quarter",
    ]

    for col in df.columns:
        if not col.startswith("weather_"):
            continue

        if any(token in col for token in engineered_tokens):
            continue

        if any(token in col.lower() for token in banned_tokens):
            continue

        base_weather_cols.append(col)

    return sorted(base_weather_cols)

def add_ptf_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_ptf = df["ptf"].shift(1)

    rolling_mean_24 = prev_ptf.rolling(24).mean()
    rolling_mean_168 = prev_ptf.rolling(168).mean()
    rolling_std_24 = prev_ptf.rolling(24).std()

    df["ptf_ratio_24"] = prev_ptf / (prev_ptf.shift(24) + EPSILON)
    df["ptf_ratio_168"] = prev_ptf / (prev_ptf.shift(168) + EPSILON)

    df["ptf_above_mean_24"] = (prev_ptf > rolling_mean_24).astype("int8")
    df["ptf_above_mean_168"] = (prev_ptf > rolling_mean_168).astype("int8")

    df["ptf_zscore_24"] = (prev_ptf - rolling_mean_24) / (rolling_std_24 + EPSILON)

    return df


def add_selected_external_regime_features(
    df: pd.DataFrame,
    jump_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()

    if "smf_smf" in df.columns:
        smf_base = df["smf_smf"].shift(1)

        smf_mean_24 = smf_base.rolling(24).mean()
        smf_mean_168 = smf_base.rolling(168).mean()

        df["smf_diff_1"] = smf_base.diff(1)
        df["smf_diff_24"] = smf_base.diff(24)
        df["smf_diff_168"] = smf_base.diff(168)

        df["smf_ratio_24"] = smf_base / (smf_base.shift(24) + EPSILON)
        df["smf_ratio_168"] = smf_base / (smf_base.shift(168) + EPSILON)

        df["smf_above_mean_24"] = (smf_base > smf_mean_24).astype("int8")
        df["smf_above_mean_168"] = (smf_base > smf_mean_168).astype("int8")

    if jump_cols is None:
        jump_cols = []

    for col in jump_cols:
        if col not in df.columns:
            continue

        base = df[col].shift(1)
        rolling_std_24 = base.rolling(24).std()
        rolling_std_168 = base.rolling(168).std()
        jump_abs = base.diff().abs()

        df[f"{col}_jump_24"] = (jump_abs > (2.5 * rolling_std_24)).astype("int8")
        df[f"{col}_jump_168"] = (jump_abs > (2.5 * rolling_std_168)).astype("int8")

    return df

def add_weather_features(df: pd.DataFrame, weather_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in weather_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        for lag in WEATHER_LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        df[f"{col}_rolling_mean_24"] = df[col].shift(1).rolling(24).mean()
        df[f"{col}_rolling_mean_168"] = df[col].shift(1).rolling(168).mean()

    return df


def get_external_feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("gen_", "cons_", "smf_")
    engineered_tokens = (
        "_lag_",
        "_rolling_mean_",
        "_rolling_std_",
        "_rolling_min_",
        "_rolling_max_",
    )

    base_cols = []
    for col in df.columns:
        if not col.startswith(prefixes):
            continue
        if any(token in col for token in engineered_tokens):
            continue
        base_cols.append(col)

    return sorted(base_cols)


def add_external_side_features(df: pd.DataFrame, external_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in external_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        base = df[col].shift(1)

        for lag in EXTERNAL_LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        for window in EXTERNAL_ROLLING_WINDOWS:
            df[f"{col}_rolling_mean_{window}"] = base.rolling(window).mean()
            df[f"{col}_rolling_std_{window}"] = base.rolling(window).std()

    return df


def add_target(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = df["ptf"].shift(-horizon)
    return df


def build_feature_list(df: pd.DataFrame) -> list[str]:
    base_features = [
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

    weather_cols = get_weather_feature_columns(df)
    weather_engineered_cols = []

    for col in weather_cols:
        weather_engineered_cols.append(col)
        for lag in WEATHER_LAGS:
            weather_engineered_cols.append(f"{col}_lag_{lag}")
        weather_engineered_cols.append(f"{col}_rolling_mean_24")
        weather_engineered_cols.append(f"{col}_rolling_mean_168")

    external_cols = get_external_feature_columns(df)
    external_engineered_cols = []

    for col in external_cols:
        external_engineered_cols.append(col)
        for lag in EXTERNAL_LAGS:
            external_engineered_cols.append(f"{col}_lag_{lag}")
        for window in EXTERNAL_ROLLING_WINDOWS:
            external_engineered_cols.append(f"{col}_rolling_mean_{window}")
            external_engineered_cols.append(f"{col}_rolling_std_{window}")

    return base_features + weather_engineered_cols + external_engineered_cols


def validate_engineered_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing_feature_cols = [col for col in feature_cols if col not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing engineered feature columns: {missing_feature_cols}")


def finalize_train_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    required_cols = feature_cols + [TARGET_COL]
    df = df.dropna(subset=required_cols).copy()
    df = df.reset_index()

    final_cols = [DATE_COL] + feature_cols + [TARGET_COL]
    return df[final_cols].copy()


def finalize_inference_latest_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=feature_cols).copy()
    if df.empty:
        raise ValueError("No valid rows found for inference_latest after feature NaN filtering.")

    df = df.tail(1).copy()
    df = df.reset_index()

    final_cols = [DATE_COL] + feature_cols
    return df[final_cols].copy()


def finalize_inference_backfill_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=feature_cols).copy()
    if df.empty:
        raise ValueError("No valid rows found for inference_backfill after feature NaN filtering.")

    df = df.reset_index()

    final_cols = [DATE_COL] + feature_cols
    return df[final_cols].copy()


def save_features(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


def run_feature_pipeline(
    mode: str,
    use_weather: bool = True,
    use_generation: bool = True,
    use_consumption: bool = True,
    use_smf: bool = True,
) -> None:
    print("Loading processed PTF data...")
    df = load_processed_data(PROCESSED_PTF_PATH)

    print("Validating required PTF columns...")
    validate_required_columns(df)

    print("Preparing base dataframe...")
    df = prepare_base_dataframe(df)

    weather_df = None
    generation_df = None
    consumption_df = None
    smf_df = None

    if use_weather:
        print("Loading processed weather data...")
        weather_df = load_weather_data(PROCESSED_WEATHER_PATH)
        print(f"Weather dataframe shape after cleaning: {weather_df.shape}")

    if use_generation:
        print("Loading processed generation data...")
        generation_df = load_generation_data(PROCESSED_GENERATION_PATH)
        print(f"Generation dataframe shape after cleaning: {generation_df.shape}")

    if use_consumption:
        print("Loading processed consumption data...")
        consumption_df = load_consumption_data(PROCESSED_CONSUMPTION_PATH)
        print(f"Consumption dataframe shape after cleaning: {consumption_df.shape}")

    if use_smf:
        print("Loading processed SMF data...")
        smf_df = load_smf_data(PROCESSED_SMF_PATH)
        print(f"SMF dataframe shape after cleaning: {smf_df.shape}")

    if any(x is not None for x in [weather_df, generation_df, consumption_df, smf_df]):
        print("Merging external data...")
        df = merge_external_features(
            df,
            weather_df=weather_df,
            generation_df=generation_df,
            consumption_df=consumption_df,
            smf_df=smf_df,
        )

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

    if mode == "train":
        print("Computing clip params from training dataframe...")
        q_low, q_high = compute_clip_bounds_from_train(df)
        save_clip_params(q_low=q_low, q_high=q_high)
        print(f"Saved clip params -> q_low={q_low:.6f}, q_high={q_high:.6f}")
    else:
        print("Loading clip params from artifacts...")
        q_low, q_high = load_clip_params()
        print(f"Loaded clip params -> q_low={q_low:.6f}, q_high={q_high:.6f}")

    print("Adding clipped robust features...")
    df = add_clipped_features(df, q_low=q_low, q_high=q_high)

    print("Adding spike features...")
    df = add_spike_features(df, q_low=q_low, q_high=q_high)
    if use_weather:
        weather_cols = get_weather_feature_columns(df)
        print(f"Adding weather features... found {len(weather_cols)} weather base columns")
        df = add_weather_features(df, weather_cols=weather_cols)

    if use_generation or use_consumption or use_smf:
        external_cols = get_external_feature_columns(df)
        print(f"Adding generation/consumption/SMF features... found {len(external_cols)} external base columns")
        df = add_external_side_features(df, external_cols=external_cols)

    feature_cols = build_feature_list(df)
    validate_engineered_features(df, feature_cols)

    if mode == "train":
        print("Adding target...")
        df = add_target(df, horizon=TARGET_HORIZON)

        print("Finalizing train feature dataset...")
        df_final = finalize_train_features(df, feature_cols)
        output_path = TRAIN_FEATURES_PATH

    elif mode == "inference_latest":
        print("Finalizing latest inference feature dataset...")
        df_final = finalize_inference_latest_features(df, feature_cols)
        output_path = INFERENCE_LATEST_PATH

    elif mode == "inference_backfill":
        print("Finalizing backfill inference feature dataset...")
        df_final = finalize_inference_backfill_features(df, feature_cols)
        output_path = INFERENCE_BACKFILL_PATH

    else:
        raise ValueError("mode must be one of: train, inference_latest, inference_backfill")

    print("Saving feature dataset...")
    save_features(df_final, output_path)

    print("Done.")
    print(f"Mode            : {mode}")
    print(f"Use weather     : {use_weather}")
    print(f"Use generation  : {use_generation}")
    print(f"Use consumption : {use_consumption}")
    print(f"Use smf         : {use_smf}")
    print(f"Final shape     : {df_final.shape}")
    print(f"Saved to        : {output_path}")

    if not df_final.empty and DATE_COL in df_final.columns:
        print(f"Date range      : {df_final[DATE_COL].min()} -> {df_final[DATE_COL].max()}")

    print("Columns:")
    for col in df_final.columns:
        print(f" - {col}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PTF features for training or inference."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "inference_latest", "inference_backfill"],
        help="Feature generation mode.",
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Disable weather feature merge.",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Disable generation feature merge.",
    )
    parser.add_argument(
        "--no-consumption",
        action="store_true",
        help="Disable consumption feature merge.",
    )
    parser.add_argument(
        "--no-smf",
        action="store_true",
        help="Disable SMF feature merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_feature_pipeline(
        mode=args.mode,
        use_weather=not args.no_weather,
        use_generation=not args.no_generation,
        use_consumption=not args.no_consumption,
        use_smf=not args.no_smf,
    )


if __name__ == "__main__":
    main()