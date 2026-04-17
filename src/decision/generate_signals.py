from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

DEFAULT_PREDICTIONS_PATH = Path("data/predictions/ptf/ptf_predictions_history.parquet")
DEFAULT_OUTPUT_PATH = Path("data/decision/ptf/ptf_decision_signals.parquet")
DEFAULT_SUMMARY_PATH = Path("data/decision/ptf/ptf_decision_summary.json")


# ============================================================
# IO helpers
# ============================================================

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return

    raise ValueError(f"Unsupported output format: {path.suffix}")


def save_json(data: dict, path: Path) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# Validation / normalization
# ============================================================

def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"forecast_time", "y_pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    if "feature_time" in out.columns:
        out["feature_time"] = pd.to_datetime(out["feature_time"], errors="coerce")

    out["forecast_time"] = pd.to_datetime(out["forecast_time"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")

    if out["forecast_time"].isna().any():
        bad = int(out["forecast_time"].isna().sum())
        raise ValueError(f"'forecast_time' contains {bad} invalid rows")

    if out["y_pred"].isna().any():
        bad = int(out["y_pred"].isna().sum())
        raise ValueError(f"'y_pred' contains {bad} invalid rows")

    if "prediction_type" not in out.columns:
        out["prediction_type"] = "unknown"

    if "model_version" not in out.columns:
        out["model_version"] = "unknown"

    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")

    return out.sort_values("forecast_time").reset_index(drop=True)


def keep_latest_prediction_per_forecast_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "created_at" in out.columns and out["created_at"].notna().any():
        out = out.sort_values(["forecast_time", "created_at"])
    else:
        out = out.sort_values(["forecast_time"])

    out = out.drop_duplicates(subset=["forecast_time"], keep="last")
    return out.reset_index(drop=True)


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
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        mask = (out["forecast_time"] >= start_ts) & (out["forecast_time"] <= end_ts)
        return out.loc[mask].sort_values("forecast_time").reset_index(drop=True)

    if mode == "latest_day":
        max_time = out["forecast_time"].max()
        target_day = max_time.normalize()
        mask = out["forecast_time"].dt.normalize() == target_day
        return out.loc[mask].sort_values("forecast_time").reset_index(drop=True)

    raise ValueError(f"Unsupported mode: {mode}")


# ============================================================
# Feature engineering for decision support
# ============================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["forecast_time"].dt.date.astype(str)
    out["hour"] = out["forecast_time"].dt.hour
    out["day_of_week"] = out["forecast_time"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def safe_qcut_three(values: pd.Series) -> pd.Series:
    """
    Robust 3-bin segmentation:
      - tries qcut
      - if too many duplicates, falls back to rank-based qcut
      - if still problematic, uses simple percentile thresholds
    """
    s = values.astype(float)

    try:
        return pd.qcut(s, q=3, labels=["low", "normal", "high"], duplicates="drop")
    except Exception:
        pass

    try:
        ranked = s.rank(method="first")
        return pd.qcut(ranked, q=3, labels=["low", "normal", "high"])
    except Exception:
        pass

    q1 = s.quantile(1 / 3)
    q2 = s.quantile(2 / 3)

    def _bucket(x: float) -> str:
        if x <= q1:
            return "low"
        if x <= q2:
            return "normal"
        return "high"

    return s.apply(_bucket).astype("category")


def add_price_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_regime"] = safe_qcut_three(out["y_pred"]).astype(str)
    return out


def add_relative_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    overall_mean = float(out["y_pred"].mean())
    overall_std = float(out["y_pred"].std(ddof=0)) if len(out) > 1 else 0.0

    out["pred_mean_horizon"] = overall_mean
    out["pred_std_horizon"] = overall_std
    out["pred_vs_mean_ratio"] = np.where(overall_mean != 0, out["y_pred"] / overall_mean, np.nan)
    out["pred_zscore_horizon"] = np.where(overall_std > 0, (out["y_pred"] - overall_mean) / overall_std, 0.0)

    return out


def add_local_volatility_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    For 24h next-day predictions, local volatility across neighboring forecasted hours
    is a useful rough proxy for intraday instability.
    """
    out = df.copy()
    out["pred_diff_1"] = out["y_pred"].diff().fillna(0.0)
    out["pred_abs_diff_1"] = out["pred_diff_1"].abs()
    out["local_volatility"] = (
        out["y_pred"]
        .rolling(window=window, min_periods=1)
        .std(ddof=0)
        .fillna(0.0)
    )
    return out


def add_risk_flags(
    df: pd.DataFrame,
    zscore_risk_threshold: float = 1.0,
    volatility_quantile: float = 0.80,
    spike_abs_diff_quantile: float = 0.90,
) -> pd.DataFrame:
    out = df.copy()

    vol_thr = float(out["local_volatility"].quantile(volatility_quantile))
    spike_thr = float(out["pred_abs_diff_1"].quantile(spike_abs_diff_quantile))

    out["is_high_price"] = (out["price_regime"] == "high").astype(int)
    out["is_low_price"] = (out["price_regime"] == "low").astype(int)
    out["is_volatile"] = (out["local_volatility"] >= vol_thr).astype(int)
    out["is_spike"] = (out["pred_abs_diff_1"] >= spike_thr).astype(int)
    out["is_extreme_price"] = (out["pred_zscore_horizon"].abs() >= zscore_risk_threshold).astype(int)

    out["risk_score"] = (
        1.5 * out["is_volatile"]
        + 1.5 * out["is_spike"]
        + 1.0 * out["is_extreme_price"]
    )

    def _risk_label(score: float) -> str:
        if score >= 3.0:
            return "high_risk"
        if score >= 1.5:
            return "medium_risk"
        return "low_risk"

    out["risk_label"] = out["risk_score"].apply(_risk_label)
    return out


def add_decision_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _signal(row: pd.Series) -> str:
        regime = row["price_regime"]
        risk = row["risk_label"]

        if regime == "high" and risk == "low_risk":
            return "HIGH_VALUE_HOUR"
        if regime == "high" and risk in {"medium_risk", "high_risk"}:
            return "HIGH_PRICE_BUT_RISKY"
        if regime == "low" and risk == "low_risk":
            return "LOW_VALUE_HOUR"
        if regime == "low" and risk in {"medium_risk", "high_risk"}:
            return "LOW_PRICE_VOLATILE"
        if regime == "normal" and risk == "high_risk":
            return "NORMAL_PRICE_RISKY"
        return "NORMAL_HOUR"

    out["decision_signal"] = out.apply(_signal, axis=1)

    def _action_text(signal: str) -> str:
        mapping = {
            "HIGH_VALUE_HOUR": "Favorable hour based on forecasted price.",
            "HIGH_PRICE_BUT_RISKY": "High forecasted price, but volatility/risk is elevated.",
            "LOW_VALUE_HOUR": "Low-value hour based on forecasted price.",
            "LOW_PRICE_VOLATILE": "Low forecasted price with elevated volatility.",
            "NORMAL_PRICE_RISKY": "Price is normal, but forecast pattern looks unstable.",
            "NORMAL_HOUR": "No strong signal.",
        }
        return mapping.get(signal, "No strong signal.")

    out["decision_note"] = out["decision_signal"].map(_action_text)
    return out


def build_decision_signals(
    df_pred: pd.DataFrame,
    local_vol_window: int = 3,
    zscore_risk_threshold: float = 1.0,
    volatility_quantile: float = 0.80,
    spike_abs_diff_quantile: float = 0.90,
) -> pd.DataFrame:
    out = df_pred.copy()
    out = add_time_features(out)
    out = add_price_regime(out)
    out = add_relative_price_features(out)
    out = add_local_volatility_features(out, window=local_vol_window)
    out = add_risk_flags(
        out,
        zscore_risk_threshold=zscore_risk_threshold,
        volatility_quantile=volatility_quantile,
        spike_abs_diff_quantile=spike_abs_diff_quantile,
    )
    out = add_decision_signal(out)

    final_cols = [
        "forecast_time",
        "date",
        "hour",
        "day_of_week",
        "is_weekend",
        "y_pred",
        "prediction_type",
        "model_version",
        "price_regime",
        "pred_mean_horizon",
        "pred_std_horizon",
        "pred_vs_mean_ratio",
        "pred_zscore_horizon",
        "pred_diff_1",
        "pred_abs_diff_1",
        "local_volatility",
        "is_high_price",
        "is_low_price",
        "is_volatile",
        "is_spike",
        "is_extreme_price",
        "risk_score",
        "risk_label",
        "decision_signal",
        "decision_note",
    ]

    existing_cols = [c for c in final_cols if c in out.columns]
    return out[existing_cols].sort_values("forecast_time").reset_index(drop=True)


# ============================================================
# Summary
# ============================================================

def build_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "row_count": 0,
            "forecast_start": None,
            "forecast_end": None,
            "avg_pred": None,
            "max_pred": None,
            "min_pred": None,
            "signal_counts": {},
            "risk_counts": {},
            "top_high_value_hours": [],
            "top_risky_hours": [],
        }

    top_high = (
        df.loc[df["decision_signal"].isin(["HIGH_VALUE_HOUR", "HIGH_PRICE_BUT_RISKY"])]
        .sort_values("y_pred", ascending=False)
        .head(5)[["forecast_time", "y_pred", "decision_signal", "risk_label"]]
        .to_dict(orient="records")
    )

    top_risky = (
        df.sort_values(["risk_score", "y_pred"], ascending=[False, False])
        .head(5)[["forecast_time", "y_pred", "decision_signal", "risk_label", "risk_score"]]
        .to_dict(orient="records")
    )

    return {
        "row_count": int(len(df)),
        "forecast_start": str(df["forecast_time"].min()),
        "forecast_end": str(df["forecast_time"].max()),
        "avg_pred": float(df["y_pred"].mean()),
        "max_pred": float(df["y_pred"].max()),
        "min_pred": float(df["y_pred"].min()),
        "signal_counts": df["decision_signal"].value_counts(dropna=False).to_dict(),
        "risk_counts": df["risk_label"].value_counts(dropna=False).to_dict(),
        "top_high_value_hours": top_high,
        "top_risky_hours": top_risky,
    }


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate decision signals from PTF forecasts.")

    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to prediction history parquet/csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to decision signals parquet/csv.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to summary json.",
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
        help="Start timestamp for mode=range, e.g. 2026-04-17 00:00:00",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="End timestamp for mode=range, e.g. 2026-04-17 23:00:00",
    )
    parser.add_argument(
        "--local-vol-window",
        type=int,
        default=3,
        help="Rolling window for local volatility over forecast horizon.",
    )
    parser.add_argument(
        "--zscore-risk-threshold",
        type=float,
        default=1.0,
        help="Absolute z-score threshold for extreme price flag.",
    )
    parser.add_argument(
        "--volatility-quantile",
        type=float,
        default=0.80,
        help="Quantile threshold for local volatility risk.",
    )
    parser.add_argument(
        "--spike-quantile",
        type=float,
        default=0.90,
        help="Quantile threshold for abs predicted hour-to-hour jump.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading predictions...")
    df_pred = load_table(args.predictions_path)

    print("Normalizing prediction columns...")
    df_pred = normalize_prediction_columns(df_pred)

    print("Keeping latest prediction per forecast_time...")
    df_pred = keep_latest_prediction_per_forecast_time(df_pred)

    print(f"Selecting horizon | mode={args.mode}")
    df_pred = select_horizon(
        df_pred,
        mode=args.mode,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    if df_pred.empty:
        raise ValueError("No predictions found after horizon selection.")

    print("Generating decision signals...")
    df_decision = build_decision_signals(
        df_pred=df_pred,
        local_vol_window=args.local_vol_window,
        zscore_risk_threshold=args.zscore_risk_threshold,
        volatility_quantile=args.volatility_quantile,
        spike_abs_diff_quantile=args.spike_quantile,
    )

    print("Building summary...")
    summary = build_summary(df_decision)

    print(f"Saving decision signals to: {args.output_path}")
    save_table(df_decision, args.output_path)

    print(f"Saving summary to: {args.summary_path}")
    save_json(summary, args.summary_path)

    print("\nDone.")
    print(df_decision.head(10).to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()