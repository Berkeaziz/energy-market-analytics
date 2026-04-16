from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/epias/smf")
PROCESSED_PATH = Path("data/processed/smf/smf_processed.parquet")


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
        "systemmarginalprice": "smf",
        "smf": "smf",
    }

    df = df.rename(columns=rename_map)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = ["date", "smf"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def resolve_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    For duplicate timestamps, prefer the row with:
    1. positive smf over zero/non-positive
    2. larger smf
    """
    df = df.copy()

    df["_smf_positive_priority"] = (df["smf"] > 0).astype("int8")
    df["_smf_sort_value"] = df["smf"].fillna(float("-inf"))

    df = df.sort_values(
        by=["date", "_smf_positive_priority", "_smf_sort_value"],
        ascending=[True, False, False],
        kind="mergesort",
    )

    df = df.drop_duplicates(subset=["date"], keep="first").copy()
    df = df.drop(columns=["_smf_positive_priority", "_smf_sort_value"])

    return df


def clean_epias_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["smf"] = pd.to_numeric(df["smf"], errors="coerce")

    drop_cols = [
        col for col in df.columns
        if col.lower() in {"hour", "year", "month", "_chunk_start", "_chunk_end"}
        or col.startswith("_chunk")
        or col.startswith("_source")
        or col.startswith("_meta")
        or col.lower().startswith("unnamed")
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df = df.dropna(subset=["date", "smf"]).copy()

    df = df.sort_values("date", kind="mergesort").copy()
    df = resolve_duplicate_timestamps(df)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def check_missing_hours(df: pd.DataFrame) -> pd.DatetimeIndex:
    full_range = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="h",
        tz=df["date"].dt.tz if hasattr(df["date"].dt, "tz") else None,
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