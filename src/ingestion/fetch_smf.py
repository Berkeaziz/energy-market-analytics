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
SMF_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/bpm/data/system-marginal-price"

OUT_DIR = "data/raw/epias/smf"
DEFAULT_START_DATE = "2018-01-01"

CHUNK_DAYS = 7
OVERLAP_DAYS = 1

MAX_RETRIES = 5
RATE_LIMIT_WAIT_SECONDS = 65
REQUEST_SLEEP_SECONDS = 2
ERROR_SLEEP_SECONDS = 5

MANUAL_RANGES: list[tuple[str, str]] | None = None

DATETIME_COLS = ["date", "datetime", "period", "periodDate", "periodTime"]


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


def fetch_smf_raw(start_date: str, end_date: str, tgt: str) -> dict:
    headers = {
        "TGT": tgt,
        "Content-Type": "application/json",
    }

    payload = {
        "startDate": f"{start_date}T00:00:00+03:00",
        "endDate": f"{end_date}T23:59:59+03:00",
    }

    r = requests.post(SMF_URL, headers=headers, json=payload, timeout=60)

    if r.status_code != 200:
        print("SMF status:", r.status_code)
        print("SMF text head:", r.text[:500])

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

    if rows is None or not isinstance(rows, list) or len(rows) == 0:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    df["_chunk_start"] = chunk_start
    df["_chunk_end"] = chunk_end

    for col in DATETIME_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(
                    df[col],
                    format="%Y-%m-%dT%H:%M:%S%z",
                    errors="coerce",
                )
            except Exception:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for c in DATETIME_COLS:
        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def write_parquet_partitioned(df: pd.DataFrame, out_dir: str | Path) -> int:
    if df is None or df.empty:
        return 0

    df = df.copy()
    out_dir = Path(out_dir)

    ts_col = detect_timestamp_column(df)

    if ts_col is not None:
        df = df.dropna(subset=[ts_col])

        if df.empty:
            return 0

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

        ts_col = detect_timestamp_column(df)
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

    end_date = pd.Timestamp.now(tz="Europe/Istanbul").strftime("%Y-%m-%d")
    return start_date, end_date


def should_fetch(start_date: str, end_date: str) -> bool:
    return pd.Timestamp(start_date) <= pd.Timestamp(end_date)


def build_fetch_ranges(
    out_dir: str | Path,
    default_start_date: str,
    overlap_days: int,
) -> list[tuple[str, str]]:
    if MANUAL_RANGES:
        print("Manual ranges mode is ACTIVE.")
        ranges: list[tuple[str, str]] = []

        for rs, re in MANUAL_RANGES:
            for s, e in daterange_chunks(rs, re, CHUNK_DAYS):
                ranges.append((s, e))

        return ranges

    start_date, end_date = resolve_fetch_window(
        out_dir=out_dir,
        default_start_date=default_start_date,
        overlap_days=overlap_days,
    )

    print(f"Resolved fetch window: {start_date} -> {end_date}")

    if not should_fetch(start_date, end_date):
        return []

    return list(daterange_chunks(start_date, end_date, CHUNK_DAYS))


def fetch_single_chunk(
    start_date: str,
    end_date: str,
    tgt: str,
    username: str,
    password: str,
) -> tuple[bool, int, str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching: {start_date} -> {end_date} | attempt={attempt}/{MAX_RETRIES}")

            raw = fetch_smf_raw(start_date, end_date, tgt)
            df = raw_to_df(raw, start_date, end_date)

            if df is None or df.empty:
                print("  No rows returned for this chunk.")
                time.sleep(REQUEST_SLEEP_SECONDS)
                return True, 0, tgt

            written = write_parquet_partitioned(df, OUT_DIR)
            print(f"  WROTE {written} rows to parquet dataset: {OUT_DIR}")

            time.sleep(REQUEST_SLEEP_SECONDS)
            return True, written, tgt

        except requests.HTTPError as ex:
            status = ex.response.status_code if ex.response is not None else None

            if status == 429:
                print(
                    f"  429 rate limit on {start_date}->{end_date}. "
                    f"Waiting {RATE_LIMIT_WAIT_SECONDS} seconds..."
                )
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue

            if status in (401, 403):
                print("  Auth issue detected. Renewing TGT...")
                tgt = get_tgt(username, password)
                time.sleep(3)
                continue

            print(f"  HTTPError on {start_date}->{end_date}: {ex}")
            time.sleep(ERROR_SLEEP_SECONDS)

        except requests.RequestException as ex:
            print(f"  RequestException on {start_date}->{end_date}: {ex}")
            time.sleep(ERROR_SLEEP_SECONDS)

        except Exception as ex:
            print(f"  Error on {start_date}->{end_date}: {ex}")
            time.sleep(ERROR_SLEEP_SECONDS)

    return False, 0, tgt


def main() -> None:
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")

    if not username or not password:
        raise RuntimeError("EPIAS_USERNAME or EPIAS_PASSWORD is missing.")

    ranges = build_fetch_ranges(
        out_dir=OUT_DIR,
        default_start_date=DEFAULT_START_DATE,
        overlap_days=OVERLAP_DAYS,
    )

    if not ranges:
        print("No new data to fetch. Exiting.")
        return

    print(f"Total chunk count: {len(ranges)}")

    tgt = get_tgt(username, password)
    print("TGT OK")

    ok_chunks = 0
    fail_chunks = 0
    total_rows = 0
    failed_ranges: list[tuple[str, str]] = []

    for s, e in ranges:
        success, written, tgt = fetch_single_chunk(
            start_date=s,
            end_date=e,
            tgt=tgt,
            username=username,
            password=password,
        )

        if success:
            ok_chunks += 1
            total_rows += written
        else:
            fail_chunks += 1
            failed_ranges.append((s, e))

    print("\nFAILED RANGES:")
    if failed_ranges:
        for fr in failed_ranges:
            print(fr)
    else:
        print("None")

    print(
        f"\nDONE. OK_CHUNKS={ok_chunks}, "
        f"FAIL_CHUNKS={fail_chunks}, "
        f"TOTAL_ROWS={total_rows}"
    )
    print(f"PARQUET DATASET: {OUT_DIR}")


if __name__ == "__main__":
    main()