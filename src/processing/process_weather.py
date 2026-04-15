from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/weather/open_meteo_hourly")
PROCESSED_PATH = Path("data/processed/weather/weather_processed.parquet")


def read_raw_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw weather parquet dataset not found: {path}")
    return pd.read_parquet(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    rename_map = {
        "time": "date",
        "timestamp": "date",
        "datetime": "date",
    }

    df = df.rename(columns=rename_map)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = ["date", "region"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["region"] = df["region"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=["date", "region"]).copy()

    # Numeric conversion for all weather columns except identifiers
    non_numeric_cols = {"date", "region", "latitude", "longitude", "elevation", "timezone"}
    weather_cols = [col for col in df.columns if col not in non_numeric_cols]

    for col in weather_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["region", "date"], kind="mergesort").reset_index(drop=True)
    return df


def resolve_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    For duplicate region-date rows, keep the row with the most non-null weather values.
    """
    df = df.copy()

    protected_cols = {"date", "region", "latitude", "longitude", "elevation", "timezone"}
    value_cols = [col for col in df.columns if col not in protected_cols]

    df["_non_null_count"] = df[value_cols].notna().sum(axis=1)

    df = df.sort_values(
        by=["region", "date", "_non_null_count"],
        ascending=[True, True, False],
        kind="mergesort",
    )

    df = df.drop_duplicates(subset=["region", "date"], keep="first").copy()
    df = df.drop(columns=["_non_null_count"])

    return df


def pivot_weather_wide(df: pd.DataFrame) -> pd.DataFrame:
    protected_cols = {"date", "region", "latitude", "longitude", "elevation", "timezone"}
    value_cols = [col for col in df.columns if col not in protected_cols]

    df_wide = df.pivot(index="date", columns="region", values=value_cols)

    df_wide.columns = [
        f"{feature}_{region}"
        for feature, region in df_wide.columns.to_flat_index()
    ]

    df_wide = df_wide.sort_index().reset_index()
    return df_wide


def check_missing_hours(df: pd.DataFrame) -> pd.DatetimeIndex:
    full_range = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="h",
    )
    missing_hours = full_range.difference(df["date"])
    return missing_hours


def save_processed_parquet(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    input_path = RAW_PATH
    output_path = PROCESSED_PATH

    df = read_raw_parquet(input_path)
    print(f"Raw weather shape: {df.shape}")

    df = standardize_columns(df)
    validate_required_columns(df)

    raw_dup_count = int(df.duplicated(subset=["date", "region"]).sum())
    print(f"Raw duplicate region-date count: {raw_dup_count}")

    df = clean_weather_data(df)
    df = resolve_duplicate_timestamps(df)

    processed_dup_count = int(df.duplicated(subset=["date", "region"]).sum())
    print(f"Processed duplicate region-date count: {processed_dup_count}")

    df_wide = pivot_weather_wide(df)
    print(f"Wide weather shape: {df_wide.shape}")

    missing_hours = check_missing_hours(df_wide)
    print(f"Missing hours count: {len(missing_hours)}")

    if len(missing_hours) > 0:
        print("First 10 missing timestamps:")
        print(missing_hours[:10])

    saved_path = save_processed_parquet(df_wide, output_path)
    print(f"Processed weather parquet saved at: {saved_path}")


if __name__ == "__main__":
    main()