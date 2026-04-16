from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/epias/consumption")
PROCESSED_PATH = Path("data/processed/consumption/consumption_processed.parquet")


def read_raw_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw parquet dataset not found: {path}")
    return pd.read_parquet(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [col.strip().lower() for col in df.columns]

    rename_map = {
        "tarih": "date",
        "datetime": "date",
        "timestamp": "date",
        "date": "date",
        "consumption": "consumption",
        "load": "consumption",
        "demand": "consumption",
        "consumptionamount": "consumption",
        "consumptıon": "consumption",  # olur da typo gelirse
    }

    df = df.rename(columns=rename_map)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = ["date", "consumption"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def resolve_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    For duplicate timestamps, prefer the row with:
    1. positive consumption over zero/non-positive
    2. larger consumption
    """
    df = df.copy()

    df["_cons_positive_priority"] = (df["consumption"] > 0).astype("int8")
    df["_cons_sort_value"] = df["consumption"].fillna(float("-inf"))

    df = df.sort_values(
        by=["date", "_cons_positive_priority", "_cons_sort_value"],
        ascending=[True, False, False],
        kind="mergesort",
    )

    df = df.drop_duplicates(subset=["date"], keep="first").copy()
    df = df.drop(columns=["_cons_positive_priority", "_cons_sort_value"])
    return df


def clean_epias_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")

    # time kolonu varsa string olarak bırak
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip()

    df = df.dropna(subset=["date", "consumption"]).copy()

    # stable dedup için önce sırala
    df = df.sort_values("date", kind="mergesort").copy()

    # duplicate timestamp çöz
    df = resolve_duplicate_timestamps(df)

    df = df.sort_values("date").reset_index(drop=True)
    return df


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
    print(f"Raw shape: {df.shape}")

    df = standardize_columns(df)
    validate_required_columns(df)

    raw_dup_count = int(pd.to_datetime(df["date"], errors="coerce").duplicated().sum())
    print(f"Raw duplicate timestamp count: {raw_dup_count}")

    df = clean_epias_data(df)
    print(f"Processed shape: {df.shape}")

    processed_dup_count = int(df["date"].duplicated().sum())
    print(f"Processed duplicate timestamp count: {processed_dup_count}")

    missing_hours = check_missing_hours(df)
    print(f"Missing hours count: {len(missing_hours)}")

    if len(missing_hours) > 0:
        print("First 10 missing timestamps:")
        print(missing_hours[:10])

    saved_path = save_processed_parquet(df, output_path)
    print(f"Processed parquet saved at: {saved_path}")


if __name__ == "__main__":
    main()