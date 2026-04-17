from pathlib import Path

import pandas as pd


PTF_PATH = Path("data/processed/ptf/ptf_processed.parquet")
SMF_PATH = Path("data/processed/smf/smf_processed.parquet")
PROCESSED_PATH = Path("data/processed/market/market_data.parquet")


def read_parquet_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
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
        "systemmarginalprice": "smf",
        "smf": "smf",
        "priceusd": "priceusd",
        "priceeur": "priceeur",
    }

    df = df.rename(columns=rename_map)
    return df


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def normalize_date_column(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Normalize all timestamps to Europe/Istanbul local time,
    then drop timezone info so both sides merge safely.
    """
    df = df.copy()

    s = pd.to_datetime(df[col], errors="coerce")

    if s.dt.tz is None:
        # naive ise local Istanbul kabul et
        s = s.dt.tz_localize("Europe/Istanbul")
    else:
        # tz-aware ise Istanbul'a çevir
        s = s.dt.tz_convert("Europe/Istanbul")

    # merge stabil olsun diye timezone bilgisini kaldır
    df[col] = s.dt.tz_localize(None)

    return df


def resolve_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    For duplicate timestamps, prefer the row with:
    1. more non-null market values
    2. positive ptf over zero/non-positive
    3. larger ptf
    4. larger smf
    """
    df = df.copy()

    value_cols = [col for col in ["ptf", "smf", "priceusd", "priceeur"] if col in df.columns]

    if not value_cols:
        df = df.sort_values("date", kind="mergesort")
        df = df.drop_duplicates(subset=["date"], keep="first").copy()
        return df

    df["_non_null_count"] = df[value_cols].notna().sum(axis=1).astype("int8")

    if "ptf" in df.columns:
        df["_ptf_positive_priority"] = (df["ptf"].fillna(0) > 0).astype("int8")
        df["_ptf_sort_value"] = df["ptf"].fillna(float("-inf"))
    else:
        df["_ptf_positive_priority"] = 0
        df["_ptf_sort_value"] = float("-inf")

    if "smf" in df.columns:
        df["_smf_sort_value"] = df["smf"].fillna(float("-inf"))
    else:
        df["_smf_sort_value"] = float("-inf")

    df = df.sort_values(
        by=[
            "date",
            "_non_null_count",
            "_ptf_positive_priority",
            "_ptf_sort_value",
            "_smf_sort_value",
        ],
        ascending=[True, False, False, False, False],
        kind="mergesort",
    )

    df = df.drop_duplicates(subset=["date"], keep="first").copy()

    df = df.drop(
        columns=[
            "_non_null_count",
            "_ptf_positive_priority",
            "_ptf_sort_value",
            "_smf_sort_value",
        ],
        errors="ignore",
    )

    return df


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = normalize_date_column(df, "date")

    if "ptf" in df.columns:
        df["ptf"] = pd.to_numeric(df["ptf"], errors="coerce")

    if "smf" in df.columns:
        df["smf"] = pd.to_numeric(df["smf"], errors="coerce")

    if "priceusd" in df.columns:
        df["priceusd"] = pd.to_numeric(df["priceusd"], errors="coerce")

    if "priceeur" in df.columns:
        df["priceeur"] = pd.to_numeric(df["priceeur"], errors="coerce")

    drop_cols = [
        col
        for col in df.columns
        if col.lower() in {"hour", "year", "month", "_chunk_start", "_chunk_end"}
        or col.startswith("_chunk")
        or col.startswith("_source")
        or col.startswith("_meta")
        or col.lower().startswith("unnamed")
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df = df.dropna(subset=["date"]).copy()

    df = df.sort_values("date", kind="mergesort").copy()
    df = resolve_duplicate_timestamps(df)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def merge_market_data(ptf_df: pd.DataFrame, smf_df: pd.DataFrame) -> pd.DataFrame:
    ptf_keep_cols = [col for col in ["date", "ptf", "priceusd", "priceeur"] if col in ptf_df.columns]
    smf_keep_cols = [col for col in ["date", "smf"] if col in smf_df.columns]

    ptf_df = ptf_df[ptf_keep_cols].copy()
    smf_df = smf_df[smf_keep_cols].copy()

    print("\nPTF date sample before merge:")
    print(ptf_df["date"].head(3))
    print("\nSMF date sample before merge:")
    print(smf_df["date"].head(3))

    print(f"\nPTF min/max: {ptf_df['date'].min()} -> {ptf_df['date'].max()}")
    print(f"SMF min/max: {smf_df['date'].min()} -> {smf_df['date'].max()}")

    common_dates = set(ptf_df["date"]).intersection(set(smf_df["date"]))
    print(f"Common timestamp count before merge: {len(common_dates)}")

    df = pd.merge(ptf_df, smf_df, on="date", how="inner")

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
    ptf_df = read_parquet_file(PTF_PATH)
    smf_df = read_parquet_file(SMF_PATH)

    print(f"PTF raw shape: {ptf_df.shape}")
    print(f"SMF raw shape: {smf_df.shape}")

    ptf_df = standardize_columns(ptf_df)
    smf_df = standardize_columns(smf_df)

    validate_required_columns(ptf_df, ["date", "ptf"])
    validate_required_columns(smf_df, ["date", "smf"])

    ptf_raw_dup_count = int(pd.to_datetime(ptf_df["date"], errors="coerce").duplicated().sum())
    smf_raw_dup_count = int(pd.to_datetime(smf_df["date"], errors="coerce").duplicated().sum())

    print(f"PTF raw duplicate timestamp count: {ptf_raw_dup_count}")
    print(f"SMF raw duplicate timestamp count: {smf_raw_dup_count}")

    ptf_df = clean_market_data(ptf_df)
    smf_df = clean_market_data(smf_df)

    print(f"PTF cleaned shape: {ptf_df.shape}")
    print(f"SMF cleaned shape: {smf_df.shape}")

    df = merge_market_data(ptf_df, smf_df)
    df = clean_market_data(df)

    print(f"Processed market shape: {df.shape}")

    processed_dup_count = int(df["date"].duplicated().sum())
    print(f"Processed market duplicate timestamp count: {processed_dup_count}")

    missing_hours = check_missing_hours(df)
    print(f"Missing hours count: {len(missing_hours)}")

    if len(missing_hours) > 0:
        print("First 10 missing timestamps:")
        print(missing_hours[:10])

    saved_path = save_processed_parquet(df, PROCESSED_PATH)
    print(f"Processed parquet saved at: {saved_path}")


if __name__ == "__main__":
    main()