from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
MCP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"

OUT_DIR = "data/raw/epias/ptf_mcp"
DEFAULT_START_DATE = "2018-01-01"
CHUNK_DAYS = 30
OVERLAP_DAYS = 1


def get_tgt(username: str, password: str) -> str:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
    }
    payload = {"username": username, "password": password}

    r = requests.post(CAS_URL, headers=headers, data=payload, timeout=30)
    r.raise_for_status()

    tgt = r.text.strip()
    if not tgt.startswith("TGT-"):
        raise RuntimeError(
            "Could not get valid TGT. "
            f"status={r.status_code} "
            f"content-type={r.headers.get('Content-Type')} "
            f"text={r.text[:300]}"
        )
    return tgt


def fetch_mcp_raw(start_date: str, end_date: str, tgt: str) -> dict:
    headers = {
        "TGT": tgt,
        "Content-Type": "application/json",
    }
    payload = {
        "startDate": f"{start_date}T00:00:00+03:00",
        "endDate": f"{end_date}T23:59:59+03:00",
    }

    r = requests.post(MCP_URL, headers=headers, json=payload, timeout=60)

    if r.status_code != 200:
        print("MCP status:", r.status_code)
        print("MCP text head:", r.text[:500])

    r.raise_for_status()
    return r.json()


def daterange_chunks(start_date: str, end_date: str, chunk_days: int):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)


def _find_first_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_first_list(v)
            if found is not None:
                return found
    return None


def raw_to_df(raw: dict, chunk_start: str, chunk_end: str) -> pd.DataFrame:
    rows = _find_first_list(raw)

    if not isinstance(rows, list) or len(rows) == 0:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    df["_chunk_start"] = chunk_start
    df["_chunk_end"] = chunk_end

    for col in ["date", "datetime", "time", "period", "periodDate", "periodTime"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    return df


def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for c in ["date", "datetime", "time", "period", "periodDate", "periodTime"]:
        if c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
    return None


def write_parquet_partitioned(df: pd.DataFrame, out_dir: str | Path) -> int:
    if df.empty:
        return 0

    df = df.copy()
    out_dir = Path(out_dir)

    ts_col = detect_timestamp_column(df)

    if ts_col is not None:
        df["year"] = df[ts_col].dt.year.astype("int16")
        df["month"] = df[ts_col].dt.month.astype("int8")
    else:
        cs = pd.to_datetime(df["_chunk_start"].iloc[0], errors="coerce")
        if pd.isna(cs):
            raise ValueError("Could not infer year/month for partitioning.")
        df["year"] = int(cs.year)
        df["month"] = int(cs.month)

    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        out_dir,
        engine="pyarrow",
        compression="snappy",
        index=False,
        partition_cols=["year", "month"],
    )

    return len(df)


def get_last_timestamp_from_parquet_dataset(path: str | Path) -> pd.Timestamp | None:
    path = Path(path)

    if not path.exists():
        return None

    try:
        df = pd.read_parquet(path)

        if df.empty:
            return None

        ts_col = None
        for c in ["date", "datetime", "time", "period", "periodDate", "periodTime"]:
            if c in df.columns:
                ts_col = c
                break

        if ts_col is None:
            return None

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])

        if df.empty:
            return None

        return df[ts_col].max()

    except Exception as e:
        print(f"WARNING: Could not read existing parquet dataset: {e}")
        return None


def resolve_fetch_window(
    out_dir: str | Path,
    default_start_date: str,
    overlap_days: int = 1,
) -> tuple[str, str]:
    last_ts = get_last_timestamp_from_parquet_dataset(out_dir)

    if last_ts is None:
        start_date = default_start_date
        print("No existing raw dataset found. Starting full backfill.")
    else:
        overlap_start = last_ts - pd.Timedelta(days=overlap_days)
        start_date = overlap_start.strftime("%Y-%m-%d")
        print(f"Last timestamp found: {last_ts}")
        print(f"Using overlap of {overlap_days} day(s).")

    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    return start_date, end_date


def should_fetch(start_date: str, end_date: str) -> bool:
    return pd.Timestamp(start_date) <= pd.Timestamp(end_date)


def main() -> None:
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")

    if not username or not password:
        raise RuntimeError("EPIAS_USERNAME ve EPIAS_PASSWORD Invalid")

    start_date, end_date = resolve_fetch_window(
        out_dir=OUT_DIR,
        default_start_date=DEFAULT_START_DATE,
        overlap_days=OVERLAP_DAYS,
    )

    print(f"Resolved fetch window: {start_date} -> {end_date}")

    if not should_fetch(start_date, end_date):
        print("No new data to fetch. Exiting.")
        return

    tgt = get_tgt(username, password)
    print("TGT OK")

    ok_chunks = 0
    fail_chunks = 0
    total_rows = 0

    for s, e in daterange_chunks(start_date, end_date, chunk_days=CHUNK_DAYS):
        try:
            print(f"Fetching: {s} -> {e}")

            raw = fetch_mcp_raw(s, e, tgt)
            df = raw_to_df(raw, s, e)

            if df.empty:
                print("  No rows returned for this chunk.")
                ok_chunks += 1
                continue

            written = write_parquet_partitioned(df, OUT_DIR)

            print(f"  WROTE {written} rows to parquet dataset: {OUT_DIR}")
            total_rows += written
            ok_chunks += 1

            time.sleep(0.2)

        except requests.HTTPError as ex:
            fail_chunks += 1
            print(f"  FAILED: {s} -> {e} | HTTPError: {ex}")
            continue

        except Exception as ex:
            fail_chunks += 1
            print(f"  FAILED: {s} -> {e} | Error: {ex}")
            continue

    print(
        f"DONE. OK_CHUNKS={ok_chunks}, "
        f"FAIL_CHUNKS={fail_chunks}, "
        f"TOTAL_ROWS={total_rows}"
    )
    print(f"PARQUET DATASET: {OUT_DIR}")


if __name__ == "__main__":
    main()