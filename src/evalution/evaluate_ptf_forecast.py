from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PREDICTION_STORE_PATH = Path("data/predictions/ptf/ptf_predictions_history.parquet")
ACTUALS_PATH = Path("data/processed/ptf/ptf_processed.parquet")

EVALUATION_DIR = Path("data/evaluation/ptf")
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_OUTPUT_PATH = EVALUATION_DIR / "ptf_prediction_evaluation.parquet"
SUMMARY_OUTPUT_PATH = EVALUATION_DIR / "ptf_prediction_evaluation_summary.json"

DATE_COL = "date"
ACTUAL_COL = "ptf"
FORECAST_COL = "forecast_time"
PRED_COL = "y_pred"

MIN_ABS_TARGET_FOR_MAPE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join prediction history with actual PTF values and compute evaluation metrics."
    )
    parser.add_argument(
        "--prediction-path",
        type=str,
        default=str(PREDICTION_STORE_PATH),
        help="Path to prediction history parquet.",
    )
    parser.add_argument(
        "--actuals-path",
        type=str,
        default=str(ACTUALS_PATH),
        help="Path to processed actuals parquet file or parquet directory.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(EVALUATION_OUTPUT_PATH),
        help="Path to output evaluation parquet.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(SUMMARY_OUTPUT_PATH),
        help="Path to output evaluation summary json.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Optional filter for a single model_version.",
    )
    parser.add_argument(
        "--prediction-type",
        type=str,
        default=None,
        choices=["latest", "backfill_range", "backfill_auto", "backfill_full"],
        help="Optional filter for a single prediction_type.",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help=(
            "Keep only the latest prediction per "
            "(forecast_time, model_version, prediction_type). "
            "Useful if old duplicates exist in the store."
        ),
    )
    return parser.parse_args()


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def to_naive_datetime(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series, errors="coerce")

    if pd.api.types.is_datetime64tz_dtype(series):
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)

    return series


def evaluate_regression(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    prefix: str = "",
) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {
            f"{prefix}mae": None,
            f"{prefix}rmse": None,
            f"{prefix}mape": None,
        }

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mape_mask = np.abs(y_true) > MIN_ABS_TARGET_FOR_MAPE
    if mape_mask.sum() == 0:
        mape = None
    else:
        mape = float(
            np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
        )

    return {
        f"{prefix}mae": float(mae),
        f"{prefix}rmse": rmse,
        f"{prefix}mape": mape,
    }


def load_prediction_history(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction history file not found: {path}")

    df = pd.read_parquet(path)

    required_cols = [
        "feature_time",
        "forecast_time",
        "y_pred",
        "prediction_type",
        "model_version",
        "run_id",
        "created_at",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Prediction history is missing required columns: {missing_cols}")

    for col in ["feature_time", "forecast_time", "created_at"]:
        df[col] = to_naive_datetime(df[col])

    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    df = df.dropna(subset=["forecast_time", "y_pred", "created_at"]).copy()
    df = df.sort_values(["forecast_time", "created_at"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Prediction history is empty after datetime/numeric cleaning.")

    return df


def load_actuals(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Actuals path not found: {path}")

    if path.is_file():
        df = pd.read_parquet(path)
    else:
        parquet_files = sorted(path.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under directory: {path}")

        frames = [pd.read_parquet(fp) for fp in parquet_files]
        df = pd.concat(frames, ignore_index=True)

    if DATE_COL not in df.columns:
        raise ValueError(f"'{DATE_COL}' column not found in actuals dataset.")
    if ACTUAL_COL not in df.columns:
        raise ValueError(f"'{ACTUAL_COL}' column not found in actuals dataset.")

    df[DATE_COL] = to_naive_datetime(df[DATE_COL])
    df[ACTUAL_COL] = pd.to_numeric(df[ACTUAL_COL], errors="coerce")

    df = df.dropna(subset=[DATE_COL, ACTUAL_COL]).copy()
    df = df.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL], keep="last")
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("Actuals dataset is empty after cleaning.")

    return df[[DATE_COL, ACTUAL_COL]].copy()


def filter_predictions(
    df: pd.DataFrame,
    model_version: str | None,
    prediction_type: str | None,
    drop_duplicates: bool,
) -> pd.DataFrame:
    filtered = df.copy()

    if model_version:
        filtered = filtered[filtered["model_version"] == model_version].copy()

    if prediction_type:
        filtered = filtered[filtered["prediction_type"] == prediction_type].copy()

    if drop_duplicates:
        filtered = (
            filtered.sort_values(["forecast_time", "created_at"])
            .drop_duplicates(
                subset=["forecast_time", "model_version", "prediction_type"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    if filtered.empty:
        raise ValueError("No prediction rows left after filtering.")

    return filtered


def build_evaluation_dataframe(
    prediction_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    actuals_for_join = actuals_df.rename(
        columns={
            DATE_COL: FORECAST_COL,
            ACTUAL_COL: "y_true",
        }
    )

    eval_df = prediction_df.merge(
        actuals_for_join,
        on=FORECAST_COL,
        how="left",
        validate="many_to_one",
    )

    eval_df["actual_available"] = eval_df["y_true"].notna()

    eval_df["error"] = eval_df["y_true"] - eval_df[PRED_COL]
    eval_df["abs_error"] = eval_df["error"].abs()
    eval_df["squared_error"] = eval_df["error"] ** 2

    ape_mask = eval_df["y_true"].notna() & (eval_df["y_true"].abs() > MIN_ABS_TARGET_FOR_MAPE)
    eval_df["ape"] = np.nan
    eval_df.loc[ape_mask, "ape"] = (
        eval_df.loc[ape_mask, "abs_error"] / eval_df.loc[ape_mask, "y_true"].abs()
    ) * 100

    eval_df["evaluation_created_at"] = pd.Timestamp.utcnow().tz_localize(None)

    preferred_cols = [
        "feature_time",
        "forecast_time",
        "y_pred",
        "y_true",
        "error",
        "abs_error",
        "squared_error",
        "ape",
        "actual_available",
        "prediction_type",
        "model_version",
        "run_id",
        "created_at",
        "evaluation_created_at",
    ]

    existing_preferred_cols = [col for col in preferred_cols if col in eval_df.columns]
    remaining_cols = [col for col in eval_df.columns if col not in existing_preferred_cols]

    eval_df = eval_df[existing_preferred_cols + remaining_cols].copy()
    eval_df = eval_df.sort_values(["forecast_time", "created_at"]).reset_index(drop=True)

    return eval_df


def build_group_metrics(
    df: pd.DataFrame,
    group_cols: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for group_keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        group_result = {
            col: value for col, value in zip(group_cols, group_keys)
        }
        group_result["rows"] = int(len(group))

        metrics = evaluate_regression(
            y_true=group["y_true"].values,
            y_pred=group["y_pred"].values,
        )
        group_result.update(metrics)
        results.append(group_result)

    sort_cols = group_cols.copy()
    results = sorted(
        results,
        key=lambda x: tuple("" if x.get(col) is None else str(x.get(col)) for col in sort_cols),
    )
    return results


def build_summary(eval_df: pd.DataFrame) -> dict[str, Any]:
    available_df = eval_df[eval_df["actual_available"]].copy()

    summary: dict[str, Any] = {
        "total_prediction_rows": int(len(eval_df)),
        "rows_with_actual": int(len(available_df)),
        "rows_without_actual": int((~eval_df["actual_available"]).sum()),
        "forecast_time_min": str(eval_df["forecast_time"].min()) if not eval_df.empty else None,
        "forecast_time_max": str(eval_df["forecast_time"].max()) if not eval_df.empty else None,
        "evaluated_forecast_time_min": (
            str(available_df["forecast_time"].min()) if not available_df.empty else None
        ),
        "evaluated_forecast_time_max": (
            str(available_df["forecast_time"].max()) if not available_df.empty else None
        ),
    }

    if available_df.empty:
        summary["overall_metrics"] = {
            "mae": None,
            "rmse": None,
            "mape": None,
        }
        summary["by_prediction_type"] = []
        summary["by_model_version"] = []
        summary["by_model_version_and_prediction_type"] = []
        return summary

    summary["overall_metrics"] = evaluate_regression(
        y_true=available_df["y_true"].values,
        y_pred=available_df["y_pred"].values,
    )

    summary["by_prediction_type"] = build_group_metrics(
        available_df,
        group_cols=["prediction_type"],
    )
    summary["by_model_version"] = build_group_metrics(
        available_df,
        group_cols=["model_version"],
    )
    summary["by_model_version_and_prediction_type"] = build_group_metrics(
        available_df,
        group_cols=["model_version", "prediction_type"],
    )

    return summary


def main() -> None:
    args = parse_args()

    prediction_path = Path(args.prediction_path)
    actuals_path = Path(args.actuals_path)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)

    print("Loading prediction history...")
    prediction_df = load_prediction_history(prediction_path)
    print(f"Prediction rows before filtering: {len(prediction_df)}")

    prediction_df = filter_predictions(
        df=prediction_df,
        model_version=args.model_version,
        prediction_type=args.prediction_type,
        drop_duplicates=args.drop_duplicates,
    )
    print(f"Prediction rows after filtering: {len(prediction_df)}")

    print("Loading actuals...")
    actuals_df = load_actuals(actuals_path)
    print(f"Actual rows: {len(actuals_df)}")

    print("Building evaluation dataframe...")
    eval_df = build_evaluation_dataframe(
        prediction_df=prediction_df,
        actuals_df=actuals_df,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    summary = build_summary(eval_df)
    save_json(summary, summary_path)

    print("Done.")
    print(f"Evaluation output path : {output_path}")
    print(f"Summary output path    : {summary_path}")
    print(f"Total prediction rows  : {summary['total_prediction_rows']}")
    print(f"Rows with actual       : {summary['rows_with_actual']}")
    print(f"Rows without actual    : {summary['rows_without_actual']}")
    print("Overall metrics:")
    print(json.dumps(summary["overall_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()