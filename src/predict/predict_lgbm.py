from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ARTIFACTS_DIR = Path("artifacts")
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_INFO_PATH = ARTIFACTS_DIR / "modelling_artifact_lgbm" /"model_info.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR /"modelling_artifact_lgbm" / "feature_columns.json"

DEFAULT_LATEST_FEATURES_PATH = Path(
    "data/features/ptf/ptf_features_inference_latest.parquet"
)
DEFAULT_BACKFILL_FEATURES_PATH = Path(
    "data/features/ptf/ptf_features_inference_backfill.parquet"
)

DATE_COL = "date"
TARGET_HORIZON_HOURS = 24


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_info_path: str | Path):
    model_info = load_json(model_info_path)

    model_path_str = model_info.get("model_path")
    if not model_path_str:
        raise KeyError(f"'model_path' not found in {model_info_path}")

    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model, model_path


def load_feature_columns(path: str | Path) -> list[str]:
    data = load_json(path)

    feature_cols = data.get("feature_columns")
    if feature_cols is None:
        raise KeyError(f"'feature_columns' not found in {path}")

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise ValueError("feature_columns must be a non-empty list.")

    return feature_cols


def resolve_input_path(mode: str, input_path: str | None) -> Path:
    if input_path:
        return Path(input_path)

    if mode == "latest":
        return DEFAULT_LATEST_FEATURES_PATH

    if mode == "backfill":
        return DEFAULT_BACKFILL_FEATURES_PATH

    raise ValueError("mode must be either 'latest' or 'backfill'")


def load_inference_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Inference features file not found: {path}")

    df = pd.read_parquet(path)

    if DATE_COL not in df.columns:
        raise ValueError(f"'{DATE_COL}' column not found in inference dataset.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df.empty:
        raise ValueError("Inference dataset is empty.")

    return df


def validate_inference_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing feature columns in inference dataset: {missing_cols}"
        )

    null_summary = df[feature_cols].isnull().sum()
    bad_cols = null_summary[null_summary > 0]
    if not bad_cols.empty:
        raise ValueError(
            "Inference dataset contains NaN values in required feature columns:\n"
            f"{bad_cols.to_string()}"
        )


def build_prediction_output(
    df: pd.DataFrame,
    preds,
    mode: str,
) -> pd.DataFrame:
    output = pd.DataFrame()

    output["feature_time"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    output["forecast_time"] = output["feature_time"] + pd.Timedelta(
        hours=TARGET_HORIZON_HOURS
    )
    output["y_pred"] = preds
    output["prediction_mode"] = mode

    optional_cols = [
        "lag_24",
        "lag_168",
        "hour",
        "day_of_week",
        "is_weekend",
    ]
    for col in optional_cols:
        if col in df.columns:
            output[col] = df[col].values

    return output


def save_predictions(df: pd.DataFrame, mode: str) -> tuple[Path, Path]:
    csv_path = PREDICTIONS_DIR / f"ptf_predictions_{mode}.csv"
    parquet_path = PREDICTIONS_DIR / f"ptf_predictions_{mode}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    return csv_path, parquet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightGBM inference for latest or backfill PTF features."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["latest", "backfill"],
        help="Inference mode.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Optional custom inference parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading model...")
    model, model_path = load_model(MODEL_INFO_PATH)
    print(f"Loaded model from: {model_path}")

    print("Loading feature columns...")
    feature_cols = load_feature_columns(FEATURE_COLUMNS_PATH)
    print(f"Expected feature count: {len(feature_cols)}")

    inference_path = resolve_input_path(args.mode, args.input_path)
    print(f"Loading inference features from: {inference_path}")
    df = load_inference_features(inference_path)
    print(f"Inference dataset shape: {df.shape}")

    print("Validating inference features...")
    validate_inference_features(df, feature_cols)

    X_inference = df[feature_cols].copy()
    print(f"Inference matrix shape: {X_inference.shape}")

    print("Generating predictions...")
    preds = model.predict(X_inference)

    prediction_df = build_prediction_output(
        df=df,
        preds=preds,
        mode=args.mode,
    )

    csv_path, parquet_path = save_predictions(prediction_df, mode=args.mode)

    print("Prediction sample:")
    print(prediction_df.head())

    print("Done.")
    print(f"Prediction rows : {len(prediction_df)}")
    print(f"Feature time min: {prediction_df['feature_time'].min()}")
    print(f"Feature time max: {prediction_df['feature_time'].max()}")
    print(f"Saved CSV       : {csv_path}")
    print(f"Saved Parquet   : {parquet_path}")


if __name__ == "__main__":
    main()