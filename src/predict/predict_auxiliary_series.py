from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.api.types import DatetimeTZDtype


DATE_COL = "date"
TARGET_HORIZON_HOURS = 24


PIPELINE_CONFIG = {
    "generation": {
        "prefix": "gen_",
        "artifact_dir": Path("artifacts/modelling_artifact_generation_lgbm"),
        "target_columns_path": Path("artifacts/modelling_artifact_generation_lgbm/target_columns.json"),
        "latest_features_path": Path("data/features/generation/generation_features_inference_latest.parquet"),
        "backfill_features_path": Path("data/features/generation/generation_features_inference_backfill.parquet"),
        "prediction_store_path": Path("data/predictions/generation/generation_predictions_history.parquet"),
        "forecast_latest_path": Path("data/forecast/generation/generation_forecast_latest.parquet"),
        "forecast_backfill_path": Path("data/forecast/generation/generation_forecast_backfill.parquet"),
    },
    "consumption": {
        "prefix": "cons_",
        "artifact_dir": Path("artifacts/modelling_artifact_consumption_lgbm"),
        "target_columns_path": Path("artifacts/modelling_artifact_consumption_lgbm/target_columns.json"),
        "latest_features_path": Path("data/features/consumption/consumption_features_inference_latest.parquet"),
        "backfill_features_path": Path("data/features/consumption/consumption_features_inference_backfill.parquet"),
        "prediction_store_path": Path("data/predictions/consumption/consumption_predictions_history.parquet"),
        "forecast_latest_path": Path("data/forecast/consumption/consumption_forecast_latest.parquet"),
        "forecast_backfill_path": Path("data/forecast/consumption/consumption_forecast_backfill.parquet"),
    },
    "smf": {
        "prefix": "smf_",
        "artifact_dir": Path("artifacts/modelling_artifact_smf_lgbm"),
        "target_columns_path": Path("artifacts/modelling_artifact_smf_lgbm/target_columns.json"),
        "latest_features_path": Path("data/features/smf/smf_features_inference_latest.parquet"),
        "backfill_features_path": Path("data/features/smf/smf_features_inference_backfill.parquet"),
        "prediction_store_path": Path("data/predictions/smf/smf_predictions_history.parquet"),
        "forecast_latest_path": Path("data/forecast/smf/smf_forecast_latest.parquet"),
        "forecast_backfill_path": Path("data/forecast/smf/smf_forecast_backfill.parquet"),
    },
}


BASE_PREDICTION_COLUMNS = [
    "feature_time",
    "forecast_time",
    "prediction_type",
    "model_version",
    "run_id",
    "created_at",
]

OPTIONAL_METADATA_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
]


def to_naive_datetime(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series, errors="coerce")
    if isinstance(series.dtype, DatetimeTZDtype):
        series = series.dt.tz_localize(None)
    return series


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_target_columns(path: str | Path) -> list[str]:
    data = load_json(path)

    target_cols = data.get("target_columns")
    if target_cols is None:
        raise KeyError(f"'target_columns' not found in {path}")

    if not isinstance(target_cols, list) or len(target_cols) == 0:
        raise ValueError("target_columns must be a non-empty list.")

    return target_cols


def load_inference_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Inference features file not found: {path}")

    df = pd.read_parquet(path)

    if DATE_COL not in df.columns:
        raise ValueError(f"'{DATE_COL}' column not found in inference dataset.")

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df[DATE_COL] = to_naive_datetime(df[DATE_COL])
    df = df.dropna(subset=[DATE_COL]).copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df.empty:
        raise ValueError("Inference dataset is empty.")

    return df


def resolve_input_path(pipeline: str, mode: str, input_path: str | None) -> Path:
    if input_path:
        return Path(input_path)

    cfg = PIPELINE_CONFIG[pipeline]

    if mode == "latest":
        return cfg["latest_features_path"]

    if mode in ["backfill_range", "backfill_auto", "backfill_full"]:
        return cfg["backfill_features_path"]

    raise ValueError(
        "mode must be one of: latest, backfill_range, backfill_auto, backfill_full"
    )


def apply_date_range_filter(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    filtered_df = df.copy()

    if start_date:
        start_ts = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df[DATE_COL] >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df[DATE_COL] <= end_ts]

    filtered_df = filtered_df.reset_index(drop=True)

    if filtered_df.empty:
        raise ValueError("No rows left after applying start_date/end_date filters.")

    return filtered_df


def normalize_prediction_store_schema(
    df: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required_cols = BASE_PREDICTION_COLUMNS + target_cols + OPTIONAL_METADATA_COLUMNS
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[required_cols].copy()

    for col in ["feature_time", "forecast_time", "created_at"]:
        df[col] = to_naive_datetime(df[col])

    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["prediction_type", "model_version", "run_id"]:
        df[col] = df[col].astype("string")

    for col in OPTIONAL_METADATA_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_prediction_store(store_path: str | Path, target_cols: list[str]) -> pd.DataFrame:
    store_path = Path(store_path)

    if not store_path.exists():
        return normalize_prediction_store_schema(
            pd.DataFrame(columns=BASE_PREDICTION_COLUMNS + target_cols),
            target_cols=target_cols,
        )

    df = pd.read_parquet(store_path)
    return normalize_prediction_store_schema(df, target_cols=target_cols)


def filter_missing_predictions_for_model_versions(
    df: pd.DataFrame,
    prediction_store_df: pd.DataFrame,
    model_versions: dict[str, str],
    target_cols: list[str],
) -> pd.DataFrame:
    candidate_df = df.copy()
    candidate_df["forecast_time"] = (
        to_naive_datetime(candidate_df[DATE_COL]) + pd.Timedelta(hours=TARGET_HORIZON_HOURS)
    )

    if prediction_store_df.empty:
        return candidate_df.drop(columns=["forecast_time"]).reset_index(drop=True)

    if not {"forecast_time", "model_version"}.issubset(prediction_store_df.columns):
        return candidate_df.drop(columns=["forecast_time"]).reset_index(drop=True)

    existing_times = None

    for target_col in target_cols:
        version = model_versions.get(target_col)
        if version is None:
            continue

        target_existing = prediction_store_df.loc[
            prediction_store_df["model_version"] == version,
            "forecast_time",
        ].dropna()

        target_existing_set = set(target_existing.tolist())

        if existing_times is None:
            existing_times = target_existing_set
        else:
            existing_times = existing_times.intersection(target_existing_set)

    if not existing_times:
        return candidate_df.drop(columns=["forecast_time"]).reset_index(drop=True)

    filtered_df = candidate_df[
        ~candidate_df["forecast_time"].isin(existing_times)
    ].copy()

    filtered_df = filtered_df.drop(columns=["forecast_time"]).reset_index(drop=True)
    return filtered_df


def build_prediction_output(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    target_cols: list[str],
    prediction_type: str,
    model_version_map: dict[str, str],
    run_id: str | None,
) -> pd.DataFrame:
    output = pd.DataFrame()
    output["feature_time"] = to_naive_datetime(df[DATE_COL])
    output["forecast_time"] = output["feature_time"] + pd.Timedelta(hours=TARGET_HORIZON_HOURS)
    output["prediction_type"] = prediction_type

    if len(set(model_version_map.values())) == 1:
        output["model_version"] = next(iter(model_version_map.values()))
    else:
        output["model_version"] = "multi_target_bundle"

    output["run_id"] = run_id if run_id else "manual"
    output["created_at"] = pd.Timestamp.now().floor("s")

    for col in target_cols:
        output[col] = preds[col].values

    for col in OPTIONAL_METADATA_COLUMNS:
        if col in df.columns:
            output[col] = df[col].values

    return normalize_prediction_store_schema(output, target_cols=target_cols)


def append_predictions_to_store(
    new_df: pd.DataFrame,
    store_path: str | Path,
    target_cols: list[str],
) -> tuple[pd.DataFrame, Path]:
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = load_prediction_store(store_path, target_cols=target_cols)
    new_df = normalize_prediction_store_schema(new_df, target_cols=target_cols)

    all_cols = BASE_PREDICTION_COLUMNS + target_cols + OPTIONAL_METADATA_COLUMNS

    existing_df = existing_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    if existing_df.empty:
        combined_df = new_df.copy()
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)

    combined_df = normalize_prediction_store_schema(combined_df, target_cols=target_cols)

    subset_required = ["feature_time", "forecast_time", "model_version", "prediction_type"]
    combined_df = combined_df.dropna(subset=subset_required)

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


def build_forecast_output(
    prediction_df: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    forecast_df = pd.DataFrame()
    forecast_df[DATE_COL] = to_naive_datetime(prediction_df["forecast_time"])

    for col in target_cols:
        forecast_df[col] = pd.to_numeric(prediction_df[col], errors="coerce")

    forecast_df = (
        forecast_df.sort_values(DATE_COL)
        .drop_duplicates(subset=[DATE_COL], keep="last")
        .reset_index(drop=True)
    )
    return forecast_df


def save_forecast_output(
    forecast_df: pd.DataFrame,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    forecast_df.to_parquet(
        path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    return path


def resolve_forecast_output_path(pipeline: str, mode: str) -> Path:
    cfg = PIPELINE_CONFIG[pipeline]

    if mode == "latest":
        return cfg["forecast_latest_path"]

    if mode in ["backfill_range", "backfill_auto", "backfill_full"]:
        return cfg["forecast_backfill_path"]

    raise ValueError("Invalid mode for forecast output path.")


def validate_inference_features_for_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing feature columns in inference dataset for target '{target_col}': {missing_cols}"
        )

    null_summary = df[feature_cols].isnull().sum()
    bad_cols = null_summary[null_summary > 0]
    if not bad_cols.empty:
        raise ValueError(
            f"Inference dataset contains NaN values in required feature columns for target '{target_col}':\n"
            f"{bad_cols.to_string()}"
        )


def load_per_target_model_metadata(
    artifact_dir: str | Path,
    target_col: str,
) -> tuple[Any, list[str], str, Path]:
    artifact_dir = Path(artifact_dir)
    target_safe = target_col.replace("/", "_")

    model_info_path = artifact_dir / f"{target_safe}_model_info.json"
    feature_columns_path = artifact_dir / f"{target_safe}_feature_columns.json"

    if not model_info_path.exists():
        raise FileNotFoundError(f"Per-target model info file not found: {model_info_path}")

    if not feature_columns_path.exists():
        raise FileNotFoundError(f"Per-target feature columns file not found: {feature_columns_path}")

    model_info = load_json(model_info_path)
    model_path_str = model_info.get("model_path")
    model_version = model_info.get("model_version")

    if not model_path_str:
        raise KeyError(f"'model_path' not found in {model_info_path}")

    if not model_version:
        raise KeyError(f"'model_version' not found in {model_info_path}")

    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    feature_json = load_json(feature_columns_path)
    feature_cols = feature_json.get("feature_columns")
    if feature_cols is None:
        raise KeyError(f"'feature_columns' not found in {feature_columns_path}")

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise ValueError(f"feature_columns must be a non-empty list in {feature_columns_path}")

    return model, feature_cols, model_version, model_path


def generate_per_target_predictions(
    df: pd.DataFrame,
    pipeline: str,
    target_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    artifact_dir = PIPELINE_CONFIG[pipeline]["artifact_dir"]

    preds_df = pd.DataFrame(index=df.index)
    model_versions: dict[str, str] = {}

    for target_col in target_cols:
        print("-" * 80)
        print(f"Loading target-specific artifacts for: {target_col}")

        model, feature_cols, model_version, model_path = load_per_target_model_metadata(
            artifact_dir=artifact_dir,
            target_col=target_col,
        )

        print(f"Loaded model: {model_path}")
        print(f"Model version: {model_version}")
        print(f"Feature count for {target_col}: {len(feature_cols)}")

        validate_inference_features_for_target(df, feature_cols, target_col)

        X_target = df[feature_cols].copy()
        target_pred = model.predict(X_target)

        if getattr(target_pred, "ndim", 1) != 1:
            raise ValueError(
                f"Expected 1D predictions for target '{target_col}', "
                f"but got shape: {getattr(target_pred, 'shape', None)}"
            )

        preds_df[target_col] = target_pred
        model_versions[target_col] = model_version

    return preds_df, model_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run auxiliary series inference for generation / consumption / smf."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=["generation", "consumption", "smf"],
        help="Auxiliary pipeline to run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["latest", "backfill_range", "backfill_auto", "backfill_full"],
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
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date for backfill_range mode. Example: 2026-04-01",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date for backfill_range mode. Example: 2026-04-03",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PIPELINE_CONFIG[args.pipeline]

    print(f"Pipeline: {args.pipeline}")

    print("Loading target columns...")
    target_cols = load_target_columns(cfg["target_columns_path"])
    print(f"Target columns loaded: {target_cols}")

    inference_path = resolve_input_path(args.pipeline, args.mode, args.input_path)
    print(f"Loading inference features from: {inference_path}")
    df = load_inference_features(inference_path)
    print(f"Inference dataset shape before filtering: {df.shape}")

    if args.mode == "latest":
        df = df.tail(1).copy()
        prediction_type = "latest"
        print(f"Using latest row only. Shape: {df.shape}")

    elif args.mode == "backfill_range":
        df = apply_date_range_filter(
            df=df,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        prediction_type = "backfill_range"
        print(f"Using range-filtered backfill rows. Shape: {df.shape}")

    elif args.mode == "backfill_auto":
        print("Loading per-target model versions for missing-row detection...")
        model_versions_for_filter: dict[str, str] = {}
        for target_col in target_cols:
            _, _, model_version, _ = load_per_target_model_metadata(
                artifact_dir=cfg["artifact_dir"],
                target_col=target_col,
            )
            model_versions_for_filter[target_col] = model_version

        prediction_store_df = load_prediction_store(
            cfg["prediction_store_path"],
            target_cols=target_cols,
        )
        print(f"Loaded prediction store rows: {len(prediction_store_df)}")

        df = filter_missing_predictions_for_model_versions(
            df=df,
            prediction_store_df=prediction_store_df,
            model_versions=model_versions_for_filter,
            target_cols=target_cols,
        )
        prediction_type = "backfill_auto"
        print(f"Using auto-detected missing rows. Shape: {df.shape}")

    elif args.mode == "backfill_full":
        prediction_type = "backfill_full"
        print(f"Running FULL backfill. Shape: {df.shape}")

    else:
        raise ValueError("Invalid mode.")

    if df.empty:
        print("No rows to predict. Exiting gracefully.")
        print(f"Pipeline            : {args.pipeline}")
        print(f"Mode                : {args.mode}")
        print("Reason              : No missing forecast rows found.")
        return

    print("Generating per-target predictions...")
    preds_df, model_version_map = generate_per_target_predictions(
        df=df,
        pipeline=args.pipeline,
        target_cols=target_cols,
    )

    prediction_df = build_prediction_output(
        df=df,
        preds=preds_df,
        target_cols=target_cols,
        prediction_type=prediction_type,
        model_version_map=model_version_map,
        run_id=args.run_id,
    )

    combined_df, store_path = append_predictions_to_store(
        new_df=prediction_df,
        store_path=cfg["prediction_store_path"],
        target_cols=target_cols,
    )

    forecast_df = build_forecast_output(
        prediction_df=prediction_df,
        target_cols=target_cols,
    )
    forecast_path = save_forecast_output(
        forecast_df=forecast_df,
        path=resolve_forecast_output_path(args.pipeline, args.mode),
    )

    print("Prediction sample:")
    print(prediction_df.head())

    print("Forecast sample:")
    print(forecast_df.head())

    print("Done.")
    print(f"Pipeline            : {args.pipeline}")
    print(f"New prediction rows : {len(prediction_df)}")
    print(f"Store total rows    : {len(combined_df)}")
    print(f"Feature time min    : {prediction_df['feature_time'].min()}")
    print(f"Feature time max    : {prediction_df['feature_time'].max()}")
    print(f"Forecast time min   : {prediction_df['forecast_time'].min()}")
    print(f"Forecast time max   : {prediction_df['forecast_time'].max()}")
    print(f"Prediction store    : {store_path}")
    print(f"Forecast parquet    : {forecast_path}")
    print(f"Prediction type     : {prediction_type}")
    print(f"Run ID              : {args.run_id if args.run_id else 'manual'}")
    print("Model versions by target:")
    for k, v in model_version_map.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()