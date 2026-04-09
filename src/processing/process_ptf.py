from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/epias/ptf_mcp")
PROCESSED_PATH = Path("data/processed/ptf/ptf_processed.parquet")


def read_raw_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [col.strip().lower() for col in df.columns]

    rename_map = {
        "tarih": "date",
        "datetime": "date",
        "timestamp": "date",
        "date": "date",
        "price": "ptf",
        "mcp": "ptf",
        "ptf": "ptf",
    }

    df = df.rename(columns=rename_map)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = ["date", "ptf"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def clean_epias_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ptf"] = pd.to_numeric(df["ptf"], errors="coerce")

    df = df.dropna(subset=["date", "ptf"])
    df = df.drop_duplicates(subset=["date"], keep="first")
    df = df.sort_values("date").reset_index(drop=True)

    return df


def check_missing_hours(df: pd.DataFrame) -> pd.DatetimeIndex:
    full_range = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="h"
    )
    missing_hours = full_range.difference(df["date"])
    return missing_hours


def save_processed_parquet(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    return output_path


def main():
    input_path = RAW_PATH
    output_path = PROCESSED_PATH

    df = read_raw_parquet(input_path)
    print(f"Raw shape: {df.shape}")

    df = standardize_columns(df)
    validate_required_columns(df)

    df = clean_epias_data(df)
    print(f"Processed shape: {df.shape}")

    missing_hours = check_missing_hours(df)
    print(f"Missing hours count: {len(missing_hours)}")

    if len(missing_hours) > 0:
        print("First 10 missing timestamps:")
        print(missing_hours[:10])

    saved_path = save_processed_parquet(df, output_path)
    print(f"Processed parquet saved at: {saved_path}")


if __name__ == "__main__":
    main()