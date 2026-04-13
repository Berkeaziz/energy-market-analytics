from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="PTF Forecast Dashboard",
    page_icon="",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "ptf" / "ptf_prediction_evaluation.parquet"
SUMMARY_PATH = PROJECT_ROOT / "data" / "evaluation" / "ptf" / "ptf_prediction_evaluation_summary.json"
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "ptf" / "ptf_predictions_history.parquet"


@st.cache_data(show_spinner=False)
def load_eval_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    df = pd.read_parquet(path)

    datetime_cols = [
        "feature_time",
        "forecast_time",
        "created_at",
        "evaluation_created_at",
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = ["y_pred", "y_true", "error", "abs_error", "squared_error", "ape"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "actual_available" in df.columns:
        df["actual_available"] = df["actual_available"].fillna(False).astype(bool)

    df = df.sort_values("forecast_time").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_prediction_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    for col in ["feature_time", "forecast_time", "created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "y_pred" in df.columns:
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    df = df.sort_values("forecast_time").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def safe_json_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    evaluated = df[df["actual_available"]].copy()
    metrics: dict[str, float | int | None] = {
        "rows": int(len(df)),
        "rows_with_actual": int(len(evaluated)),
        "mae": None,
        "rmse": None,
        "mape": None,
        "bias": None,
    }

    if evaluated.empty:
        return metrics

    metrics["mae"] = float(evaluated["abs_error"].mean())
    metrics["rmse"] = float(np.sqrt(evaluated["squared_error"].mean()))
    metrics["bias"] = float(evaluated["error"].mean())

    nonzero_ape = evaluated["ape"].dropna()
    metrics["mape"] = float(nonzero_ape.mean()) if not nonzero_ape.empty else None
    return metrics


def latest_prediction_table(pred_df: pd.DataFrame, limit: int = 24) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    latest_created_at = pred_df["created_at"].dropna().max()
    if pd.isna(latest_created_at):
        return pred_df.tail(limit).copy()

    latest_df = pred_df[pred_df["created_at"] == latest_created_at].copy()
    latest_df = latest_df.sort_values("forecast_time").tail(limit)

    cols = [
        col
        for col in [
            "forecast_time",
            "y_pred",
            "prediction_type",
            "model_version",
            "run_id",
            "created_at",
        ]
        if col in latest_df.columns
    ]
    return latest_df[cols].copy()


st.title("PTF Forecast Dashboard")
st.caption("EPİAŞ PTF prediction, evaluation and forecast monitoring dashboard")

try:
    eval_df = load_eval_data(EVAL_PATH)
    pred_df = load_prediction_data(PRED_PATH)
    summary = safe_json_summary(SUMMARY_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Filters")

    model_versions = sorted(eval_df["model_version"].dropna().unique().tolist())
    prediction_types = sorted(eval_df["prediction_type"].dropna().unique().tolist())

    selected_models = st.multiselect(
        "Model version",
        options=model_versions,
        default=model_versions[-1:] if model_versions else [],
    )

    selected_types = st.multiselect(
        "Prediction type",
        options=prediction_types,
        default=prediction_types,
    )

    only_actuals = st.checkbox("Only rows with actuals", value=True)

    min_time = eval_df["forecast_time"].min()
    max_time = eval_df["forecast_time"].max()

    date_range = st.date_input(
        "Forecast date range",
        value=(min_time.date(), max_time.date()) if pd.notna(min_time) and pd.notna(max_time) else None,
    )

    chart_window = st.selectbox(
        "Chart window",
        options=[24, 72, 168, 336, 720],
        index=2,
        format_func=lambda x: f"Last {x} hours",
    )

filtered_df = eval_df.copy()

if selected_models:
    filtered_df = filtered_df[filtered_df["model_version"].isin(selected_models)].copy()

if selected_types:
    filtered_df = filtered_df[filtered_df["prediction_type"].isin(selected_types)].copy()

if only_actuals:
    filtered_df = filtered_df[filtered_df["actual_available"]].copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    forecast_tz = filtered_df["forecast_time"].dt.tz

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if forecast_tz is not None:
        start_date = start_date.tz_localize(forecast_tz)
        end_date = end_date.tz_localize(forecast_tz)

    filtered_df = filtered_df[
        (filtered_df["forecast_time"] >= start_date)
        & (filtered_df["forecast_time"] <= end_date)
    ].copy()

filtered_df = filtered_df.sort_values("forecast_time").reset_index(drop=True)
metrics = compute_metrics(filtered_df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{metrics['rows']:,}")
c2.metric("Rows with actual", f"{metrics['rows_with_actual']:,}")
c3.metric("MAE", "-" if metrics["mae"] is None else f"{metrics['mae']:.2f}")
c4.metric("RMSE", "-" if metrics["rmse"] is None else f"{metrics['rmse']:.2f}")
c5.metric("Bias", "-" if metrics["bias"] is None else f"{metrics['bias']:.2f}")

if metrics["mape"] is not None:
    st.caption(
        f"MAPE: {metrics['mape']:.2f}% — note that this metric can be unstable for zero or near-zero electricity prices."
    )

chart_df = filtered_df.tail(chart_window).copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "Actual vs Prediction",
    "Error Analysis",
    "Latest Forecast",
    "Data Preview",
])

with tab1:
    st.subheader("Actual vs Prediction")
    if chart_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        plot_df = chart_df[["forecast_time", "y_pred", "y_true"]].copy()
        plot_df = plot_df.rename(
            columns={
                "forecast_time": "time",
                "y_pred": "Prediction",
                "y_true": "Actual",
            }
        ).set_index("time")
        st.line_chart(plot_df)

        st.markdown("**Recent evaluated rows**")
        preview_cols = [
            col
            for col in [
                "forecast_time",
                "y_pred",
                "y_true",
                "error",
                "abs_error",
                "prediction_type",
                "model_version",
            ]
            if col in chart_df.columns
        ]
        st.dataframe(chart_df[preview_cols].tail(50), use_container_width=True)

with tab2:
    st.subheader("Error Analysis")
    if filtered_df.empty:
        st.warning("No error data available for the selected filters.")
    else:
        error_view = filtered_df[filtered_df["actual_available"]].copy()

        if error_view.empty:
            st.info("There are no rows with actual values in the selected range.")
        else:
            hourly_error = (
                error_view.groupby(error_view["forecast_time"].dt.hour)["abs_error"]
                .mean()
                .reset_index()
                .rename(columns={"forecast_time": "hour", "abs_error": "mean_abs_error"})
            )
            hourly_error = hourly_error.rename(columns={hourly_error.columns[0]: "hour"})
            st.markdown("**Mean absolute error by forecast hour**")
            st.bar_chart(hourly_error.set_index("hour"))

            daily_mae = (
                error_view.groupby(error_view["forecast_time"].dt.date)["abs_error"]
                .mean()
                .reset_index()
                .rename(columns={"forecast_time": "date", "abs_error": "daily_mae"})
            )
            daily_mae = daily_mae.rename(columns={daily_mae.columns[0]: "date"})
            st.markdown("**Daily MAE**")
            st.line_chart(daily_mae.set_index("date"))

            worst_rows = error_view.sort_values("abs_error", ascending=False).head(20)
            worst_cols = [
                col
                for col in [
                    "forecast_time",
                    "y_pred",
                    "y_true",
                    "error",
                    "abs_error",
                    "prediction_type",
                    "model_version",
                ]
                if col in worst_rows.columns
            ]
            st.markdown("**Top 20 worst forecast errors**")
            st.dataframe(worst_rows[worst_cols], use_container_width=True)

with tab3:
    st.subheader("Latest Forecast Snapshot")
    latest_df = latest_prediction_table(pred_df, limit=24)

    if latest_df.empty:
        st.info("Prediction history file is empty or latest forecast could not be derived.")
    else:
        st.dataframe(latest_df, use_container_width=True)

        latest_chart = latest_df.copy()
        if "forecast_time" in latest_chart.columns and "y_pred" in latest_chart.columns:
            latest_chart = latest_chart.rename(
                columns={"forecast_time": "time", "y_pred": "Prediction"}
            ).set_index("time")
            st.markdown("**Latest prediction curve**")
            st.line_chart(latest_chart[["Prediction"]])

with tab4:
    st.subheader("Data Preview")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Evaluation summary**")
        st.json(summary if summary else {"info": "summary json not found"})

    with col_b:
        st.markdown("**Filtered dataframe shape**")
        st.write(filtered_df.shape)
        st.markdown("**Available columns**")
        st.write(filtered_df.columns.tolist())

    st.markdown("**Filtered evaluation rows**")
    st.dataframe(filtered_df.tail(200), use_container_width=True)
