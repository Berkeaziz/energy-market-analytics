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


FEATURES_PATH = Path("data/features/smf/smf_features.parquet")

ARTIFACTS_DIR = Path("artifacts/modelling_artifact_smf_lgbm")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "date"

TARGET_BASE_COL = "smf_smf"
TARGET_COLUMNS = [TARGET_BASE_COL]

TARGET_COL = f"{TARGET_BASE_COL}_target"
BASELINE_COL = f"{TARGET_BASE_COL}_lag_24"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EXPERIMENT_NAME = "smf_forecasting_lgbm"

DROP_COLS = [
    DATE_COL,
    TARGET_COL,
]


DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 700,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.3,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


# =========================================================
# IO
# =========================================================
def load_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_parquet(path).copy()

    if DATE_COL not in df.columns:
        raise ValueError(f"'{DATE_COL}' column not found in dataset: {path}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# VALIDATION
# =========================================================
def validate_dataset(df: pd.DataFrame) -> None:
    required_cols = [DATE_COL, TARGET_COL, BASELINE_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def validate_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns in dataset: {missing}")


# =========================================================
# SPLIT
# =========================================================
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
        raise ValueError("One of the split datasets is empty. Check dataset size and split ratios.")

    return train_df, val_df, test_df


# =========================================================
# SAMPLE WEIGHT
# =========================================================
def compute_recency_weights(
    dates: pd.Series,
    half_life_years: float = 2.5,
    min_weight: float = 0.25,
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


# =========================================================
# FEATURE SELECTION
# =========================================================
def build_smf_feature_list(df: pd.DataFrame, target_base_col: str = TARGET_BASE_COL) -> list[str]:
    time_features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    candidate_features = [
        target_base_col,
        f"{target_base_col}_lag_1",
        f"{target_base_col}_lag_24",
        f"{target_base_col}_lag_48",
        f"{target_base_col}_lag_72",
        f"{target_base_col}_lag_168",
        f"{target_base_col}_rolling_mean_24",
        f"{target_base_col}_rolling_std_24",
        f"{target_base_col}_rolling_min_24",
        f"{target_base_col}_rolling_max_24",
        f"{target_base_col}_rolling_mean_168",
        f"{target_base_col}_rolling_std_168",
        f"{target_base_col}_rolling_min_168",
        f"{target_base_col}_rolling_max_168",
        f"{target_base_col}_diff_1",
        f"{target_base_col}_diff_24",
        f"{target_base_col}_ratio_24",
        f"{target_base_col}_ratio_168",
    ]

    extra_feature_patterns = (
        "_lag_24",
        "_lag_168",
        "_rolling_mean_24",
        "_rolling_std_24",
        "_rolling_mean_168",
        "_rolling_std_168",
    )

    extra_candidates: list[str] = []
    for col in df.columns:
        if col in time_features or col in candidate_features:
            continue
        if col in DROP_COLS:
            continue
        if not col.startswith("smf_"):
            continue
        if col == TARGET_COL:
            continue
        if col.endswith("_target"):
            continue
        if any(col.endswith(pattern) for pattern in extra_feature_patterns):
            extra_candidates.append(col)

    selected = []
    for col in time_features + candidate_features + sorted(extra_candidates):
        if col in df.columns and col not in selected:
            selected.append(col)

    if not selected:
        raise ValueError("No feature columns selected for SMF model.")

    return selected


# =========================================================
# METRICS
# =========================================================
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
                    (y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]
                )
            ) * 100
        )

    return {
        f"{prefix}mae": float(mae),
        f"{prefix}rmse": float(rmse),
        f"{prefix}mape": mape,
    }


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("Loading SMF feature dataset...")
    df = load_features(FEATURES_PATH)
    validate_dataset(df)

    target_columns_path = ARTIFACTS_DIR / "target_columns.json"
    save_json({"target_columns": TARGET_COLUMNS}, target_columns_path)

    print(f"Dataset shape before cleaning: {df.shape}")
    df = df.dropna(subset=[TARGET_COL, BASELINE_COL]).copy()
    print(f"Dataset shape after target/baseline cleaning: {df.shape}")

    print("\nSelecting features...")
    feature_cols = build_smf_feature_list(df, target_base_col=TARGET_BASE_COL)
    validate_feature_columns(df, feature_cols)

    print(f"Selected feature count: {len(feature_cols)}")
    print("Selected features:")
    print(feature_cols)

    required_for_model = feature_cols + [TARGET_COL, BASELINE_COL]
    df = df.dropna(subset=required_for_model).copy()
    print(f"\nDataset shape after feature NaN cleaning: {df.shape}")

    train_df, val_df, test_df = chronological_split(df)
    train_val_df = pd.concat([train_df, val_df], axis=0).copy()

    train_val_weights = compute_recency_weights(
        train_val_df[DATE_COL],
        half_life_years=2.5,
        min_weight=0.25,
        max_weight=1.00,
    )

    print("\nSample weight summary (train+val):")
    print(
        f"min={train_val_weights.min():.4f}, "
        f"max={train_val_weights.max():.4f}, "
        f"mean={train_val_weights.mean():.4f}"
    )

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
        "target_base_col": TARGET_BASE_COL,
        "target_col": TARGET_COL,
        "baseline_col": BASELINE_COL,
        "feature_count": int(len(feature_cols)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_val_rows": int(len(train_val_df)),
        "test_rows": int(len(test_df)),
        **baseline_test_metrics,
    }

    print("Baseline test metrics:")
    print(json.dumps(baseline_metrics, indent=2, ensure_ascii=False))

    print("\nConfiguring MLflow...")
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_version = f"{TARGET_BASE_COL}_smf_lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=f"lightgbm_final_train_val_{TARGET_BASE_COL}"):
        mlflow.set_tag("model_version", model_version)
        mlflow.set_tag("pipeline", "smf")
        mlflow.set_tag("target_base_col", TARGET_BASE_COL)
        mlflow.set_tag("target_col", TARGET_COL)

        mlflow.log_param("model_type", "LGBMRegressor")
        mlflow.log_param("target_base_col", TARGET_BASE_COL)
        mlflow.log_param("target_col", TARGET_COL)
        mlflow.log_param("baseline_col", BASELINE_COL)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("train_ratio", TRAIN_RATIO)
        mlflow.log_param("val_ratio", VAL_RATIO)
        mlflow.log_param("test_ratio", TEST_RATIO)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("sample_weighting", "recency_exponential")
        mlflow.log_param("sample_weight_half_life_years", 2.5)
        mlflow.log_param("sample_weight_min", 0.25)
        mlflow.log_param("sample_weight_max", 1.00)

        mlflow.log_params(DEFAULT_PARAMS)

        mlflow.log_metric("train_rows", int(len(train_df)))
        mlflow.log_metric("val_rows", int(len(val_df)))
        mlflow.log_metric("train_val_rows", int(len(train_val_df)))
        mlflow.log_metric("test_rows", int(len(test_df)))

        mlflow.log_metric("baseline_test_mae", baseline_metrics["test_mae"])
        mlflow.log_metric("baseline_test_rmse", baseline_metrics["test_rmse"])
        if baseline_metrics["test_mape"] is not None:
            mlflow.log_metric("baseline_test_mape", baseline_metrics["test_mape"])

        print("\nTraining final LightGBM model on train + val...")
        model = LGBMRegressor(**DEFAULT_PARAMS)
        model.fit(
            X_train_val,
            y_train_val,
            sample_weight=train_val_weights,
        )

        test_pred = model.predict(X_test)

        lgbm_test_metrics = evaluate_regression(y_test, test_pred, prefix="test_")
        lgbm_metrics = {
            "model": "lightgbm_final_train_val_smf",
            "model_version": model_version,
            "target_base_col": TARGET_BASE_COL,
            "target_col": TARGET_COL,
            "baseline_col": BASELINE_COL,
            "feature_count": int(len(feature_cols)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_val_rows": int(len(train_val_df)),
            "test_rows": int(len(test_df)),
            **lgbm_test_metrics,
        }

        print("LightGBM test metrics:")
        print(json.dumps(lgbm_metrics, indent=2, ensure_ascii=False))

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

        print("\nTop feature importances:")
        print(importance_df.head(30))

        print("\nSaving artifacts...")
        target_safe = TARGET_BASE_COL.replace("/", "_")

        model_path = ARTIFACTS_DIR / f"{target_safe}_lgbm_model.joblib"
        model_info_path = ARTIFACTS_DIR / f"{target_safe}_model_info.json"
        feature_cols_path = ARTIFACTS_DIR / f"{target_safe}_feature_columns.json"
        baseline_metrics_path = ARTIFACTS_DIR / f"{target_safe}_baseline_test_metrics.json"
        lgbm_metrics_path = ARTIFACTS_DIR / f"{target_safe}_lgbm_test_metrics.json"
        best_params_path = ARTIFACTS_DIR / f"{target_safe}_best_params_used.json"
        feature_importance_path = ARTIFACTS_DIR / f"{target_safe}_feature_importance.csv"
        test_predictions_path = ARTIFACTS_DIR / f"{target_safe}_test_predictions.csv"
        split_summary_path = ARTIFACTS_DIR / f"{target_safe}_split_summary.json"
        metrics_comparison_path = ARTIFACTS_DIR / f"{target_safe}_metrics_comparison.csv"

        joblib.dump(model, model_path)

        save_json(
            {
                "model_path": model_path.as_posix(),
                "model_version": model_version,
                "target_base_col": TARGET_BASE_COL,
                "target_col": TARGET_COL,
                "baseline_col": BASELINE_COL,
                "feature_count": len(feature_cols),
            },
            model_info_path,
        )

        save_json({"feature_columns": feature_cols}, feature_cols_path)
        save_json(baseline_metrics, baseline_metrics_path)
        save_json(lgbm_metrics, lgbm_metrics_path)
        save_json(DEFAULT_PARAMS, best_params_path)

        comparison_df.to_csv(metrics_comparison_path, index=False)
        importance_df.to_csv(feature_importance_path, index=False)

        test_pred_df = pd.DataFrame(
            {
                DATE_COL: test_df[DATE_COL].values,
                "y_true": y_test.values,
                "y_pred": test_pred,
                "baseline_pred": test_baseline_pred,
            }
        )
        test_pred_df.to_csv(test_predictions_path, index=False)

        split_summary = {
            "train_start": str(train_df[DATE_COL].min()),
            "train_end": str(train_df[DATE_COL].max()),
            "val_start": str(val_df[DATE_COL].min()),
            "val_end": str(val_df[DATE_COL].max()),
            "test_start": str(test_df[DATE_COL].min()),
            "test_end": str(test_df[DATE_COL].max()),
        }
        save_json(split_summary, split_summary_path)

        mlflow.log_artifact(str(target_columns_path))
        mlflow.log_artifact(str(model_info_path))
        mlflow.log_artifact(str(baseline_metrics_path))
        mlflow.log_artifact(str(lgbm_metrics_path))
        mlflow.log_artifact(str(feature_cols_path))
        mlflow.log_artifact(str(best_params_path))
        mlflow.log_artifact(str(split_summary_path))
        mlflow.log_artifact(str(metrics_comparison_path))
        mlflow.log_artifact(str(feature_importance_path))
        mlflow.log_artifact(str(test_predictions_path))
        mlflow.log_artifact(str(model_path))

        print("Done.")
        print(f"Artifacts saved to: {ARTIFACTS_DIR.resolve()}")
        print(f"Model version: {model_version}")


if __name__ == "__main__":
    main()