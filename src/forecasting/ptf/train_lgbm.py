from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES_PATH = Path("data/features/ptf/ptf_features.parquet")
ARTIFACTS_DIR = Path("artifacts/modelling_artifact_lgbm")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
DATE_COL = "date"
BASELINE_COL = "lag_24"

DROP_COLS = [
    TARGET_COL,
    DATE_COL,
]

BEST_PARAMS_PATH = Path("artifacts/modelling_artifact/ptf_best_params_optuna.json")
BEST_FEATURES_PATH = Path("artifacts/modelling_artifact/ptf_feature_columns.json")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EXPERIMENT_NAME = "ptf_forecasting_lgbm"


def load_best_params(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Best params file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_best_features(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Best features file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        best_features = json.load(f)

    if not isinstance(best_features, list) or not best_features:
        raise ValueError("Best features file must contain a non-empty JSON list.")

    return best_features

def compute_recency_weights(
    dates: pd.Series,
    half_life_years: float = 2.5,
    min_weight: float = 0.20,
    max_weight: float = 1.00,
) -> np.ndarray:
    dates = pd.to_datetime(dates, errors="coerce")

    if dates.isna().any():
        raise ValueError("DATE_COL contains invalid datetimes; cannot compute sample weights.")

    max_date = dates.max()
    age_days = (max_date - dates).dt.days.astype(float)

    half_life_days = 365.25 * half_life_years
    decay_lambda = np.log(2.0) / half_life_days

    weights = np.exp(-decay_lambda * age_days)
    weights = np.clip(weights, min_weight, max_weight)

    return weights.astype(np.float32)

def load_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_parquet(path)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    required_cols = [TARGET_COL, BASELINE_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def validate_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing_feature_cols = [col for col in feature_cols if col not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing selected feature columns in dataset: {missing_feature_cols}")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(df)
    if total == 0:
        raise ValueError("Input dataframe is empty.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "One of the split datasets is empty. Check dataset size and split ratios."
        )

    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, drop_cols: list[str]) -> list[str]:
    feature_cols = [col for col in df.columns if col not in drop_cols]

    if not feature_cols:
        raise ValueError("No feature columns found after dropping excluded columns.")

    return feature_cols


def evaluate_regression(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    prefix: str = "",
) -> dict[str, float | None]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    nonzero_mask = y_true != 0
    if nonzero_mask.sum() == 0:
        mape = None
    else:
        mape = float(
            np.mean(
                np.abs(
                    (y_true[nonzero_mask] - y_pred[nonzero_mask])
                    / y_true[nonzero_mask]
                )
            ) * 100
        )

    return {
        f"{prefix}mae": float(mae),
        f"{prefix}rmse": float(rmse),
        f"{prefix}mape": mape,
    }


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    print("Loading features...")
    df = load_features(FEATURES_PATH)
    validate_dataset(df)

    print(f"Dataset shape before cleaning: {df.shape}")
    df = df.dropna(subset=[TARGET_COL, BASELINE_COL]).copy()
    print(f"Dataset shape after target/baseline cleaning: {df.shape}")

    print("\nLoading best features (TOP 30)...")
    feature_cols = load_best_features(BEST_FEATURES_PATH)
    validate_feature_columns(df, feature_cols)

    print(f"Selected feature count: {len(feature_cols)}")
    print("Selected features:")
    print(feature_cols)

    all_available_features = get_feature_columns(df, DROP_COLS)
    print(f"\nTotal available feature count in dataset: {len(all_available_features)}")
    print("First 15 available features:")
    print(all_available_features[:15])

    train_df, val_df, test_df = chronological_split(df)
    train_val_df = pd.concat([train_df, val_df], axis=0).copy()

    train_val_weights = compute_recency_weights(
        train_val_df[DATE_COL],
        half_life_years=2.5,
        min_weight=0.20,
        max_weight=1.00,
    )

    print("\nSample weight summary (train+val):")
    print(f"min={train_val_weights.min():.4f}, max={train_val_weights.max():.4f}, mean={train_val_weights.mean():.4f}")

    X_train_val = train_val_df[feature_cols].copy()
    y_train_val = train_val_df[TARGET_COL].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    print("\nSplit shapes:")
    print(f"Train     : {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Train+Val : {X_train_val.shape}, {y_train_val.shape}")
    print(f"Test      : {X_test.shape}, {y_test.shape}")

    print("\nRunning baseline evaluation on test set...")
    test_baseline_pred = test_df[BASELINE_COL].values

    baseline_test_metrics = evaluate_regression(
        y_test,
        test_baseline_pred,
        prefix="test_",
    )

    baseline_metrics = {
        "model": "baseline_lag_24",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_val_rows": int(len(train_val_df)),
        "test_rows": int(len(test_df)),
        **baseline_test_metrics,
    }

    print("Baseline test metrics:")
    print(json.dumps(baseline_metrics, indent=2))

    print("\nLoading best params...")
    best_params = load_best_params(BEST_PARAMS_PATH)
    print(json.dumps(best_params, indent=2))

    print("\nConfiguring MLflow...")
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_version = f"lgbm_top30_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name="lightgbm_final_train_val_top30"):
        mlflow.set_tag("model_version", model_version)
        mlflow.log_param("sample_weighting", "recency_exponential")
        mlflow.log_param("sample_weight_half_life_years", 2.5)
        mlflow.log_param("sample_weight_min", 0.20)
        mlflow.log_param("sample_weight_max", 1.00)
        mlflow.log_param("model_type", "LGBMRegressor")
        mlflow.log_param("target_col", TARGET_COL)
        mlflow.log_param("baseline_col", BASELINE_COL)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("train_ratio", TRAIN_RATIO)
        mlflow.log_param("val_ratio", VAL_RATIO)
        mlflow.log_param("test_ratio", TEST_RATIO)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("feature_source", BEST_FEATURES_PATH.as_posix())

        mlflow.log_params(best_params)

        mlflow.log_metric("train_rows", int(len(train_df)))
        mlflow.log_metric("val_rows", int(len(val_df)))
        mlflow.log_metric("train_val_rows", int(len(train_val_df)))
        mlflow.log_metric("test_rows", int(len(test_df)))

        mlflow.log_metric("baseline_test_mae", baseline_metrics["test_mae"])
        mlflow.log_metric("baseline_test_rmse", baseline_metrics["test_rmse"])
        if baseline_metrics["test_mape"] is not None:
            mlflow.log_metric("baseline_test_mape", baseline_metrics["test_mape"])

        print("\nTraining final LightGBM model on train + val with TOP 30 features...")
        model = LGBMRegressor(**best_params)
        model.fit(
            X_train_val,
            y_train_val,
            sample_weight=train_val_weights,
        )

        test_pred = model.predict(X_test)

        lgbm_test_metrics = evaluate_regression(y_test, test_pred, prefix="test_")

        lgbm_metrics = {
            "model": "lightgbm_final_train_val_top30",
            "model_version": model_version,
            "feature_count": int(len(feature_cols)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_val_rows": int(len(train_val_df)),
            "test_rows": int(len(test_df)),
            **lgbm_test_metrics,
        }

        print("LightGBM test metrics:")
        print(json.dumps(lgbm_metrics, indent=2))

        mlflow.log_metric("test_mae", lgbm_metrics["test_mae"])
        mlflow.log_metric("test_rmse", lgbm_metrics["test_rmse"])
        if lgbm_metrics["test_mape"] is not None:
            mlflow.log_metric("test_mape", lgbm_metrics["test_mape"])

        comparison_df = pd.DataFrame([baseline_metrics, lgbm_metrics])
        print("\nModel comparison:")
        print(comparison_df)

        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 30 feature importances (selected set):")
        print(importance_df.head(30))

        print("\nSaving artifacts...")

        model_path = ARTIFACTS_DIR / "lgbm_model.joblib"
        model_info_path = ARTIFACTS_DIR / "model_info.json"

        joblib.dump(model, model_path)

        save_json(
            {
                "model_path": model_path.as_posix(),
                "model_version": model_version,
                "feature_count": len(feature_cols),
                "best_features_path": BEST_FEATURES_PATH.as_posix(),
            },
            model_info_path,
        )

        comparison_df.to_csv(ARTIFACTS_DIR / "metrics_comparison.csv", index=False)
        importance_df.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)

        save_json(baseline_metrics, ARTIFACTS_DIR / "baseline_test_metrics.json")
        save_json(lgbm_metrics, ARTIFACTS_DIR / "lgbm_test_metrics.json")
        save_json({"feature_columns": feature_cols}, ARTIFACTS_DIR / "feature_columns.json")
        save_json(best_params, ARTIFACTS_DIR / "best_params_used.json")

        test_pred_df = pd.DataFrame(
            {
                "y_true": y_test.values,
                "y_pred": test_pred,
                "baseline_pred": test_baseline_pred,
            }
        )

        if DATE_COL in test_df.columns:
            test_pred_df.insert(0, DATE_COL, test_df[DATE_COL].values)

        test_pred_df.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

        split_summary = {
            "train_start": str(train_df[DATE_COL].min()) if DATE_COL in train_df.columns else None,
            "train_end": str(train_df[DATE_COL].max()) if DATE_COL in train_df.columns else None,
            "val_start": str(val_df[DATE_COL].min()) if DATE_COL in val_df.columns else None,
            "val_end": str(val_df[DATE_COL].max()) if DATE_COL in val_df.columns else None,
            "test_start": str(test_df[DATE_COL].min()) if DATE_COL in test_df.columns else None,
            "test_end": str(test_df[DATE_COL].max()) if DATE_COL in test_df.columns else None,
        }
        save_json(split_summary, ARTIFACTS_DIR / "split_summary.json")

        mlflow.log_artifact(str(BEST_FEATURES_PATH))
        mlflow.log_artifact(str(model_info_path))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "baseline_test_metrics.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "lgbm_test_metrics.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "feature_columns.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "best_params_used.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "split_summary.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "metrics_comparison.csv"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "feature_importance.csv"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "test_predictions.csv"))
        mlflow.log_artifact(str(model_path))

        print("Done.")
        print(f"Artifacts saved to: {ARTIFACTS_DIR.resolve()}")
        print(f"Saved model path in JSON: {model_path.as_posix()}")
        print(f"Model version: {model_version}")


if __name__ == "__main__":
    main()