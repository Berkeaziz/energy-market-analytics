from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

OUT_DIR = "data/raw/weather/open_meteo_hourly"
DEFAULT_START_DATE = "2018-01-01"

CHUNK_DAYS = 30
OVERLAP_DAYS = 1

MAX_RETRIES = 5
REQUEST_SLEEP_SECONDS = 1.5
ERROR_SLEEP_SECONDS = 5

TIMEZONE = "Europe/Istanbul"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]

REGIONS = {
    "marmara": {"latitude": 41.0082, "longitude": 28.9784},
    "ege": {"latitude": 38.4237, "longitude": 27.1428},
    "ic_anadolu": {"latitude": 39.9334, "longitude": 32.8597},
    "akdeniz": {"latitude": 36.8969, "longitude": 30.7133},
    "karadeniz": {"latitude": 41.2867, "longitude": 36.3300},
    "guneydogu": {"latitude": 37.0662, "longitude": 37.3833},
}


def daterange_chunks(start_date: str, end_date: str, chunk_days: int):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)


def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for c in ["timestamp", "date", "datetime", "time"]:
        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def write_parquet_partitioned(df: pd.DataFrame, out_dir: str | Path) -> int:
    if df.empty:
        return 0

    df = df.copy()
    out_dir = Path(out_dir)

    ts_col = detect_timestamp_column(df)
    if ts_col is None:
        raise ValueError("Timestamp column not found.")

    df = df.dropna(subset=[ts_col])
    if df.empty:
        return 0

    df["year"] = df[ts_col].dt.year.astype("int16")
    df["month"] = df[ts_col].dt.month.astype("int8")

    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        out_dir,
        engine="pyarrow",
        compression="snappy",
        index=False,
        partition_cols=["region", "year", "month"],
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
        print("No existing weather dataset found. Starting full backfill.")
    else:
        overlap_start = last_ts - pd.Timedelta(days=overlap_days)
        start_date = overlap_start.strftime("%Y-%m-%d")
        print(f"Last timestamp found: {last_ts}")
        print(f"Using overlap of {overlap_days} day(s).")

    end_date = pd.Timestamp.now(tz=TIMEZONE).strftime("%Y-%m-%d")
    return start_date, end_date


def should_fetch(start_date: str, end_date: str) -> bool:
    return pd.Timestamp(start_date) <= pd.Timestamp(end_date)


def build_fetch_ranges(
    out_dir: str | Path,
    default_start_date: str,
    overlap_days: int,
) -> list[tuple[str, str]]:
    start_date, end_date = resolve_fetch_window(
        out_dir=out_dir,
        default_start_date=default_start_date,
        overlap_days=overlap_days,
    )

    print(f"Resolved fetch window: {start_date} -> {end_date}")

    if not should_fetch(start_date, end_date):
        return []

    return list(daterange_chunks(start_date, end_date, CHUNK_DAYS))


def fetch_weather_raw(start_date: str, end_date: str) -> dict | list:
    latitudes = ",".join(str(v["latitude"]) for v in REGIONS.values())
    longitudes = ",".join(str(v["longitude"]) for v in REGIONS.values())

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }

    r = requests.get(BASE_URL, params=params, timeout=60)

    if r.status_code != 200:
        print("Weather status:", r.status_code)
        print("Weather text head:", r.text[:500])

    r.raise_for_status()
    return r.json()


def normalize_single_location_payload(payload: dict, region_name: str) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["region"] = region_name
    return df


def raw_to_df(raw: dict | list, chunk_start: str, chunk_end: str) -> pd.DataFrame:
    region_names = list(REGIONS.keys())

    if isinstance(raw, dict):
        frames = [normalize_single_location_payload(raw, region_names[0])]
    else:
        frames = [
            normalize_single_location_payload(item, region_name)
            for item, region_name in zip(raw, region_names)
        ]

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        return df

    df["_chunk_start"] = chunk_start
    df["_chunk_end"] = chunk_end
    return df


def fetch_single_chunk(start_date: str, end_date: str) -> tuple[bool, int]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching: {start_date} -> {end_date} | attempt={attempt}/{MAX_RETRIES}")

            raw = fetch_weather_raw(start_date, end_date)
            df = raw_to_df(raw, start_date, end_date)

            if df.empty:
                print("  No rows returned for this chunk.")
                time.sleep(REQUEST_SLEEP_SECONDS)
                return True, 0

            written = write_parquet_partitioned(df, OUT_DIR)
            print(f"  WROTE {written} rows to parquet dataset: {OUT_DIR}")

            time.sleep(REQUEST_SLEEP_SECONDS)
            return True, written

        except requests.HTTPError as ex:
            status = ex.response.status_code if ex.response is not None else None

            if status == 429:
                print("  429 rate limit. Waiting 60 seconds...")
                time.sleep(60)
                continue

            print(f"  HTTPError on {start_date}->{end_date}: {ex}")
            time.sleep(ERROR_SLEEP_SECONDS)

        except requests.RequestException as ex:
            print(f"  RequestException on {start_date}->{end_date}: {ex}")
            time.sleep(ERROR_SLEEP_SECONDS)

        except Exception as ex:
            print(f"  Error on {start_date}->{end_date}: {ex}")
            time.sleep(ERROR_SLEEP_SECONDS)

    return False, 0


def main() -> None:
    ranges = build_fetch_ranges(
        out_dir=OUT_DIR,
        default_start_date=DEFAULT_START_DATE,
        overlap_days=OVERLAP_DAYS,
    )

    if not ranges:
        print("No new weather data to fetch. Exiting.")
        return

    print(f"Total chunk count: {len(ranges)}")

    ok_chunks = 0
    fail_chunks = 0
    total_rows = 0
    failed_ranges: list[tuple[str, str]] = []

    for s, e in ranges:
        success, written = fetch_single_chunk(start_date=s, end_date=e)

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