from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Default paths
# ============================================================

DEFAULT_DECISION_PATH = Path("data/decision/ptf/ptf_decision_signals.parquet")
DEFAULT_OUTPUT_PATH = Path("data/decision/ptf/ptf_strategy_simulation.parquet")
DEFAULT_SUMMARY_PATH = Path("data/decision/ptf/ptf_strategy_simulation_summary.json")


# ============================================================
# Generic helpers
# ============================================================

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return

    raise ValueError(f"Unsupported output format: {path.suffix}")


def save_json(data: dict, path: Path) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# Datetime helpers
# ============================================================

def to_naive_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a datetime-like series to pandas datetime and strip timezone info.
    Result dtype becomes datetime64[ns].
    """
    s = pd.to_datetime(series, errors="coerce")

    # timezone-aware ise naive yap
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert(None)

    # bazı edge case'lerde object dönebilir; elemana göre kontrol edelim
    try:
        return s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return s


def normalize_timestamp_value(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


# ============================================================
# Column detection / normalization
# ============================================================

def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "forecast_time",
        "datetime",
        "date",
        "timestamp",
        "time",
        "period",
        "periodDate",
        "periodTime",
    ]
    return find_first_existing(df, candidates)


def detect_ptf_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "ptf",
        "PTF",
        "price",
        "marketClearingPrice",
        "mcp",
        "actual_ptf",
    ]
    return find_first_existing(df, candidates)


def detect_smf_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "smf",
        "SMF",
        "systemMarginalPrice",
        "system_marginal_price",
    ]
    return find_first_existing(df, candidates)


def detect_generation_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "generation",
        "actual_generation",
        "actual_generation_mwh",
        "generation_mwh",
        "production",
        "production_mwh",
    ]
    return find_first_existing(df, candidates)


def normalize_decision_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"forecast_time", "y_pred", "decision_signal", "risk_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Decision file missing required columns: {sorted(missing)}")

    out = df.copy()
    out["forecast_time"] = to_naive_datetime(out["forecast_time"])
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")

    if out["forecast_time"].isna().any():
        raise ValueError("Invalid forecast_time values in decision file.")
    if out["y_pred"].isna().any():
        raise ValueError("Invalid y_pred values in decision file.")

    out = out.drop_duplicates(subset=["forecast_time"], keep="last")
    return out.sort_values("forecast_time").reset_index(drop=True)


def normalize_market_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    time_col = detect_time_column(out)
    if time_col is None:
        raise ValueError("Could not detect time column in market file.")

    ptf_col = detect_ptf_column(out)
    smf_col = detect_smf_column(out)

    if ptf_col is None and smf_col is None:
        raise ValueError("Could not detect either PTF or SMF column in market file.")

    out = out.rename(columns={time_col: "forecast_time"})

    if ptf_col is not None and ptf_col != "ptf":
        out = out.rename(columns={ptf_col: "ptf"})
    if smf_col is not None and smf_col != "smf":
        out = out.rename(columns={smf_col: "smf"})

    out["forecast_time"] = to_naive_datetime(out["forecast_time"])

    if "ptf" in out.columns:
        out["ptf"] = pd.to_numeric(out["ptf"], errors="coerce")
    if "smf" in out.columns:
        out["smf"] = pd.to_numeric(out["smf"], errors="coerce")

    out = out.dropna(subset=["forecast_time"]).copy()
    out = out.drop_duplicates(subset=["forecast_time"], keep="last")
    return out.sort_values("forecast_time").reset_index(drop=True)


def normalize_generation_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    time_col = detect_time_column(out)
    if time_col is None:
        raise ValueError("Could not detect time column in generation file.")

    gen_col = detect_generation_column(out)
    if gen_col is None:
        raise ValueError("Could not detect generation column in generation file.")

    out = out.rename(columns={time_col: "forecast_time", gen_col: "actual_generation_mwh"})
    out["forecast_time"] = to_naive_datetime(out["forecast_time"])
    out["actual_generation_mwh"] = pd.to_numeric(out["actual_generation_mwh"], errors="coerce")

    out = out.dropna(subset=["forecast_time"]).copy()
    out = out.drop_duplicates(subset=["forecast_time"], keep="last")
    return out.sort_values("forecast_time").reset_index(drop=True)


# ============================================================
# Selection helpers
# ============================================================

def select_horizon(
    df: pd.DataFrame,
    mode: str = "latest_day",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()

    if mode == "all":
        return out.reset_index(drop=True)

    if mode == "range":
        if start_time is None or end_time is None:
            raise ValueError("For mode='range', both start_time and end_time must be provided.")

        start_ts = normalize_timestamp_value(start_time)
        end_ts = normalize_timestamp_value(end_time)

        mask = (out["forecast_time"] >= start_ts) & (out["forecast_time"] <= end_ts)
        return out.loc[mask].sort_values("forecast_time").reset_index(drop=True)

    if mode == "latest_day":
        target_day = out["forecast_time"].max().normalize()
        mask = out["forecast_time"].dt.normalize() == target_day
        return out.loc[mask].sort_values("forecast_time").reset_index(drop=True)

    raise ValueError(f"Unsupported mode: {mode}")


# ============================================================
# Strategy logic
# ============================================================

def add_bid_strategies(
    df: pd.DataFrame,
    base_generation: float,
    high_multiplier: float,
    low_multiplier: float,
    risky_high_multiplier: float,
    risky_low_multiplier: float,
    normal_multiplier: float,
    min_bid_mwh: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()

    out["baseline_bid_mwh"] = float(base_generation)

    signal_to_multiplier = {
        "HIGH_VALUE_HOUR": high_multiplier,
        "HIGH_PRICE_BUT_RISKY": risky_high_multiplier,
        "LOW_VALUE_HOUR": low_multiplier,
        "LOW_PRICE_VOLATILE": risky_low_multiplier,
        "NORMAL_PRICE_RISKY": normal_multiplier,
        "NORMAL_HOUR": normal_multiplier,
    }

    out["strategy_multiplier"] = out["decision_signal"].map(signal_to_multiplier).fillna(normal_multiplier)
    out["strategy_bid_mwh"] = (base_generation * out["strategy_multiplier"]).clip(lower=min_bid_mwh)

    return out


def add_actual_generation(
    df: pd.DataFrame,
    generation_df: Optional[pd.DataFrame],
    base_generation: float,
    simulate_generation_noise_std: float,
    random_seed: int,
) -> pd.DataFrame:
    out = df.copy()

    out["forecast_time"] = to_naive_datetime(out["forecast_time"])

    if generation_df is not None:
        generation_df = generation_df.copy()
        generation_df["forecast_time"] = to_naive_datetime(generation_df["forecast_time"])

        out = out.merge(
            generation_df[["forecast_time", "actual_generation_mwh"]],
            on="forecast_time",
            how="left",
        )
    else:
        out["actual_generation_mwh"] = np.nan

    if out["actual_generation_mwh"].isna().all():
        rng = np.random.default_rng(random_seed)
        noise = rng.normal(loc=0.0, scale=simulate_generation_noise_std, size=len(out))
        simulated = base_generation * (1.0 + noise)
        out["actual_generation_mwh"] = np.clip(simulated, a_min=0.0, a_max=None)

    out["actual_generation_mwh"] = out["actual_generation_mwh"].fillna(base_generation)

    return out


def add_market_prices(
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    use_pred_as_ptf_fallback: bool,
    smf_from_ptf_multiplier: float,
) -> pd.DataFrame:
    out = df.copy()
    out["forecast_time"] = to_naive_datetime(out["forecast_time"])

    if market_df is not None:
        market_df = market_df.copy()
        market_df["forecast_time"] = to_naive_datetime(market_df["forecast_time"])

        cols = ["forecast_time"]
        if "ptf" in market_df.columns:
            cols.append("ptf")
        if "smf" in market_df.columns:
            cols.append("smf")

        market_part = market_df[cols].drop_duplicates(subset=["forecast_time"], keep="last")
        out = out.merge(market_part, on="forecast_time", how="left")
    else:
        out["ptf"] = np.nan
        out["smf"] = np.nan

    if "ptf" not in out.columns:
        out["ptf"] = np.nan

    if use_pred_as_ptf_fallback:
        out["ptf"] = out["ptf"].fillna(out["y_pred"])

    if "smf" not in out.columns:
        out["smf"] = np.nan

    out["smf"] = out["smf"].fillna(out["ptf"] * smf_from_ptf_multiplier)

    if out["ptf"].isna().any():
        raise ValueError("PTF could not be resolved. Provide market file or enable y_pred fallback.")
    if out["smf"].isna().any():
        raise ValueError("SMF could not be resolved. Provide market file or valid SMF fallback settings.")

    return out


# ============================================================
# Settlement / revenue calculations
# ============================================================

def calculate_strategy_financials(
    bid_col: str,
    actual_col: str,
    ptf_col: str,
    smf_col: str,
    prefix: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    imbalance_col = f"{prefix}_imbalance_mwh"
    dayahead_revenue_col = f"{prefix}_dayahead_revenue"
    imbalance_cashflow_col = f"{prefix}_imbalance_cashflow"
    total_revenue_col = f"{prefix}_total_revenue"

    out[imbalance_col] = out[actual_col] - out[bid_col]
    out[dayahead_revenue_col] = out[bid_col] * out[ptf_col]
    out[imbalance_cashflow_col] = out[imbalance_col] * out[smf_col]
    out[total_revenue_col] = out[dayahead_revenue_col] + out[imbalance_cashflow_col]

    return out


def add_financial_comparison(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = calculate_strategy_financials(
        bid_col="baseline_bid_mwh",
        actual_col="actual_generation_mwh",
        ptf_col="ptf",
        smf_col="smf",
        prefix="baseline",
        df=out,
    )

    out = calculate_strategy_financials(
        bid_col="strategy_bid_mwh",
        actual_col="actual_generation_mwh",
        ptf_col="ptf",
        smf_col="smf",
        prefix="strategy",
        df=out,
    )

    out["delta_total_revenue"] = out["strategy_total_revenue"] - out["baseline_total_revenue"]
    out["delta_dayahead_revenue"] = out["strategy_dayahead_revenue"] - out["baseline_dayahead_revenue"]
    out["delta_imbalance_cashflow"] = out["strategy_imbalance_cashflow"] - out["baseline_imbalance_cashflow"]

    out["baseline_abs_imbalance_mwh"] = out["baseline_imbalance_mwh"].abs()
    out["strategy_abs_imbalance_mwh"] = out["strategy_imbalance_mwh"].abs()
    out["delta_abs_imbalance_mwh"] = out["strategy_abs_imbalance_mwh"] - out["baseline_abs_imbalance_mwh"]

    return out


# ============================================================
# Summary
# ============================================================

def build_summary(df: pd.DataFrame, base_generation: float) -> dict:
    if df.empty:
        return {
            "row_count": 0,
            "base_generation": base_generation,
        }

    best_hours_df = (
        df.sort_values("delta_total_revenue", ascending=False)
        .head(5)[
            [
                "forecast_time",
                "y_pred",
                "ptf",
                "smf",
                "decision_signal",
                "baseline_bid_mwh",
                "strategy_bid_mwh",
                "actual_generation_mwh",
                "delta_total_revenue",
            ]
        ]
        .copy()
    )
    best_hours_df["forecast_time"] = best_hours_df["forecast_time"].astype(str)

    worst_hours_df = (
        df.sort_values("delta_total_revenue", ascending=True)
        .head(5)[
            [
                "forecast_time",
                "y_pred",
                "ptf",
                "smf",
                "decision_signal",
                "baseline_bid_mwh",
                "strategy_bid_mwh",
                "actual_generation_mwh",
                "delta_total_revenue",
            ]
        ]
        .copy()
    )
    worst_hours_df["forecast_time"] = worst_hours_df["forecast_time"].astype(str)

    total_baseline_revenue = float(df["baseline_total_revenue"].sum())
    total_strategy_revenue = float(df["strategy_total_revenue"].sum())
    total_gain = float(df["delta_total_revenue"].sum())

    baseline_abs_imb = float(df["baseline_abs_imbalance_mwh"].sum())
    strategy_abs_imb = float(df["strategy_abs_imbalance_mwh"].sum())

    gain_pct_vs_baseline = (
        (total_gain / total_baseline_revenue) * 100.0
        if total_baseline_revenue != 0
        else None
    )

    return {
        "row_count": int(len(df)),
        "forecast_start": str(df["forecast_time"].min()),
        "forecast_end": str(df["forecast_time"].max()),
        "base_generation_mwh": float(base_generation),
        "total_baseline_revenue": total_baseline_revenue,
        "total_strategy_revenue": total_strategy_revenue,
        "total_gain": total_gain,
        "gain_pct_vs_baseline": gain_pct_vs_baseline,
        "total_baseline_abs_imbalance_mwh": baseline_abs_imb,
        "total_strategy_abs_imbalance_mwh": strategy_abs_imb,
        "imbalance_reduction_mwh": baseline_abs_imb - strategy_abs_imb,
        "signal_counts": df["decision_signal"].value_counts(dropna=False).to_dict(),
        "best_hours": best_hours_df.to_dict(orient="records"),
        "worst_hours": worst_hours_df.to_dict(orient="records"),
    }


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate baseline vs signal-based bidding strategy.")

    parser.add_argument(
        "--decision-path",
        type=Path,
        default=DEFAULT_DECISION_PATH,
        help="Path to decision signals parquet/csv.",
    )
    parser.add_argument(
        "--market-path",
        type=Path,
        default=None,
        help="Optional market file containing actual PTF and/or SMF.",
    )
    parser.add_argument(
        "--generation-path",
        type=Path,
        default=None,
        help="Optional generation file containing actual generation per hour.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output strategy simulation parquet/csv.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to output summary json.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="latest_day",
        choices=["latest_day", "range", "all"],
        help="Selection mode for forecast horizon.",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start timestamp for range mode.",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="End timestamp for range mode.",
    )

    parser.add_argument(
        "--base-generation",
        type=float,
        default=50.0,
        help="Baseline planned generation / bid in MWh.",
    )
    parser.add_argument(
        "--high-multiplier",
        type=float,
        default=1.20,
        help="Bid multiplier for HIGH_VALUE_HOUR.",
    )
    parser.add_argument(
        "--low-multiplier",
        type=float,
        default=0.80,
        help="Bid multiplier for LOW_VALUE_HOUR.",
    )
    parser.add_argument(
        "--risky-high-multiplier",
        type=float,
        default=1.05,
        help="Bid multiplier for HIGH_PRICE_BUT_RISKY.",
    )
    parser.add_argument(
        "--risky-low-multiplier",
        type=float,
        default=0.90,
        help="Bid multiplier for LOW_PRICE_VOLATILE.",
    )
    parser.add_argument(
        "--normal-multiplier",
        type=float,
        default=1.00,
        help="Bid multiplier for NORMAL_HOUR and NORMAL_PRICE_RISKY.",
    )
    parser.add_argument(
        "--min-bid-mwh",
        type=float,
        default=0.0,
        help="Lower bound for strategy bid.",
    )

    parser.add_argument(
        "--use-pred-as-ptf-fallback",
        action="store_true",
        help="If actual PTF is missing, use y_pred as fallback.",
    )
    parser.add_argument(
        "--smf-from-ptf-multiplier",
        type=float,
        default=1.00,
        help="Fallback SMF = PTF * multiplier when SMF is missing.",
    )

    parser.add_argument(
        "--simulate-generation-noise-std",
        type=float,
        default=0.10,
        help="Std dev for simulated generation noise ratio if actual generation file is missing.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for simulated generation fallback.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading decision signals...")
    df = load_table(args.decision_path)
    df = normalize_decision_df(df)

    print(f"Selecting horizon | mode={args.mode}")
    df = select_horizon(
        df,
        mode=args.mode,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    if df.empty:
        raise ValueError("No rows found after decision horizon selection.")

    market_df = None
    if args.market_path is not None:
        print(f"Loading market file from: {args.market_path}")
        market_df = load_table(args.market_path)
        market_df = normalize_market_df(market_df)

    generation_df = None
    if args.generation_path is not None:
        print(f"Loading generation file from: {args.generation_path}")
        generation_df = load_table(args.generation_path)
        generation_df = normalize_generation_df(generation_df)

    print("Adding market prices...")
    df = add_market_prices(
        df=df,
        market_df=market_df,
        use_pred_as_ptf_fallback=args.use_pred_as_ptf_fallback,
        smf_from_ptf_multiplier=args.smf_from_ptf_multiplier,
    )

    print("Adding baseline and strategy bids...")
    df = add_bid_strategies(
        df=df,
        base_generation=args.base_generation,
        high_multiplier=args.high_multiplier,
        low_multiplier=args.low_multiplier,
        risky_high_multiplier=args.risky_high_multiplier,
        risky_low_multiplier=args.risky_low_multiplier,
        normal_multiplier=args.normal_multiplier,
        min_bid_mwh=args.min_bid_mwh,
    )

    print("Adding actual generation...")
    df = add_actual_generation(
        df=df,
        generation_df=generation_df,
        base_generation=args.base_generation,
        simulate_generation_noise_std=args.simulate_generation_noise_std,
        random_seed=args.random_seed,
    )

    print("Calculating financial comparison...")
    df = add_financial_comparison(df)

    output_cols = [
        "forecast_time",
        "y_pred",
        "ptf",
        "smf",
        "decision_signal",
        "risk_label",
        "baseline_bid_mwh",
        "strategy_multiplier",
        "strategy_bid_mwh",
        "actual_generation_mwh",
        "baseline_imbalance_mwh",
        "strategy_imbalance_mwh",
        "baseline_dayahead_revenue",
        "strategy_dayahead_revenue",
        "baseline_imbalance_cashflow",
        "strategy_imbalance_cashflow",
        "baseline_total_revenue",
        "strategy_total_revenue",
        "delta_dayahead_revenue",
        "delta_imbalance_cashflow",
        "delta_total_revenue",
        "baseline_abs_imbalance_mwh",
        "strategy_abs_imbalance_mwh",
        "delta_abs_imbalance_mwh",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].copy()

    print("Building summary...")
    summary = build_summary(df_out, base_generation=args.base_generation)

    print(f"Saving simulation output to: {args.output_path}")
    save_table(df_out, args.output_path)

    print(f"Saving summary to: {args.summary_path}")
    save_json(summary, args.summary_path)

    print("\nDone.")
    print(df_out.head(10).to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()