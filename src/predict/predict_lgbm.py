from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ARTIFACTS_DIR = Path("artifacts")
MODEL_INFO_PATH = ARTIFACTS_DIR / "modelling_artifact_lgbm" / "model_info.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "modelling_artifact_lgbm" / "feature_columns.json"

DEFAULT_LATEST_FEATURES_PATH = Path("data/features/ptf/ptf_features_inference_latest.parquet")
DEFAULT_BACKFILL_FEATURES_PATH = Path("data/features/ptf/ptf_features_inference_backfill.parquet")

PREDICTION_STORE_PATH = Path("data/predictions/ptf/ptf_predictions_history.parquet")
PREDICTION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATE_COL = "date"
TARGET_HORIZON_HOURS = 24


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_metadata(model_info_path: str | Path) -> tuple[Any, Path, str]:
    model_info = load_json(model_info_path)

    model_path_str = model_info.get("model_path")
    if not model_path_str:
        raise KeyError(f"'model_path' not found in {model_info_path}")

    model_version = model_info.get("model_version")
    if not model_version:
        raise KeyError(f"'model_version' not found in {model_info_path}")

    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model, model_path, model_version


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
        raise ValueError(f"Missing feature columns in inference dataset: {missing_cols}")

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
    model_version: str,
    run_id: str | None,
) -> pd.DataFrame:
    output = pd.DataFrame()

    output["feature_time"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    output["forecast_time"] = output["feature_time"] + pd.Timedelta(hours=TARGET_HORIZON_HOURS)
    output["y_pred"] = preds
    output["prediction_type"] = mode
    output["model_version"] = model_version
    output["run_id"] = run_id if run_id else "manual"
    output["created_at"] = pd.Timestamp.utcnow()

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


def append_predictions_to_store(
    new_df: pd.DataFrame,
    store_path: str | Path,
) -> tuple[pd.DataFrame, Path]:
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if store_path.exists():
        existing_df = pd.read_parquet(store_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df.copy()

    combined_df["feature_time"] = pd.to_datetime(combined_df["feature_time"], errors="coerce")
    combined_df["forecast_time"] = pd.to_datetime(combined_df["forecast_time"], errors="coerce")
    combined_df["created_at"] = pd.to_datetime(combined_df["created_at"], errors="coerce")

    combined_df = combined_df.sort_values(
        by=["forecast_time", "created_at"]
    ).drop_duplicates(
        subset=["forecast_time", "model_version", "prediction_type"],
        keep="last",
    ).reset_index(drop=True)

    combined_df.to_parquet(
        store_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    return combined_df, store_path


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
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional pipeline run identifier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading model...")
    model, model_path, model_version = load_model_metadata(MODEL_INFO_PATH)
    print(f"Loaded model from: {model_path}")
    print(f"Model version: {model_version}")

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
        model_version=model_version,
        run_id=args.run_id,
    )

    combined_df, store_path = append_predictions_to_store(
        new_df=prediction_df,
        store_path=PREDICTION_STORE_PATH,
    )

    print("Prediction sample:")
    print(prediction_df.head())

    print("Done.")
    print(f"New prediction rows : {len(prediction_df)}")
    print(f"Store total rows    : {len(combined_df)}")
    print(f"Feature time min    : {prediction_df['feature_time'].min()}")
    print(f"Feature time max    : {prediction_df['feature_time'].max()}")
    print(f"Prediction store    : {store_path}")
    print(f"Model version       : {model_version}")
    print(f"Run ID              : {args.run_id if args.run_id else 'manual'}")


if __name__ == "__main__":
    main()