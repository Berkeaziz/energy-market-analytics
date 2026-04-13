from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

    numeric_cols = [
        "y_pred",
        "y_true",
        "error",
        "abs_error",
        "squared_error",
        "ape",
        "lag_24",
        "lag_168",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "actual_available" in df.columns:
        df["actual_available"] = df["actual_available"].fillna(False).astype(bool)

    return df.sort_values("forecast_time").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_prediction_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)

    for col in ["feature_time", "forecast_time", "created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = ["y_pred", "lag_24", "lag_168"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("forecast_time").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def safe_json_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_to_latest_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()

    if "created_at" in df.columns and df["created_at"].notna().any():
        latest_ts = df["created_at"].dropna().max()
        return df[df["created_at"] == latest_ts].copy()

    if "evaluation_created_at" in df.columns and df["evaluation_created_at"].notna().any():
        latest_ts = df["evaluation_created_at"].dropna().max()
        return df[df["evaluation_created_at"] == latest_ts].copy()

    return df.copy()


def deduplicate_forecast_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "forecast_time" not in df.columns:
        return df.copy()

    df = df.copy()

    sort_cols: list[str] = ["forecast_time"]
    if "created_at" in df.columns:
        sort_cols.append("created_at")
    elif "evaluation_created_at" in df.columns:
        sort_cols.append("evaluation_created_at")

    df = df.sort_values(sort_cols)
    df = df.drop_duplicates(subset=["forecast_time"], keep="last")
    return df.sort_values("forecast_time").reset_index(drop=True)


def apply_date_filter(df: pd.DataFrame, date_range) -> pd.DataFrame:
    if df.empty or "forecast_time" not in df.columns:
        return df.copy()

    if not (isinstance(date_range, tuple) and len(date_range) == 2):
        return df.copy()

    out = df.copy()

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    tz_info = None
    if "forecast_time" in out.columns and pd.api.types.is_datetime64_any_dtype(out["forecast_time"]):
        try:
            tz_info = out["forecast_time"].dt.tz
        except Exception:
            tz_info = None

    if tz_info is not None:
        if start_date.tzinfo is None:
            start_date = start_date.tz_localize(tz_info)
        else:
            start_date = start_date.tz_convert(tz_info)

        if end_date.tzinfo is None:
            end_date = end_date.tz_localize(tz_info)
        else:
            end_date = end_date.tz_convert(tz_info)

    return out[
        (out["forecast_time"] >= start_date) &
        (out["forecast_time"] <= end_date)
    ].copy()


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    evaluated = df[df["actual_available"]].copy() if "actual_available" in df.columns else pd.DataFrame()

    metrics: dict[str, float | int | None] = {
        "rows": int(len(df)),
        "rows_with_actual": int(len(evaluated)),
        "mae": None,
        "rmse": None,
        "mape": None,
        "bias": None,
        "future_rows": int((~df["actual_available"]).sum()) if "actual_available" in df.columns else 0,
    }

    if evaluated.empty:
        return metrics

    if "abs_error" in evaluated.columns:
        metrics["mae"] = float(evaluated["abs_error"].mean())

    if "squared_error" in evaluated.columns:
        metrics["rmse"] = float(np.sqrt(evaluated["squared_error"].mean()))

    if "error" in evaluated.columns:
        metrics["bias"] = float(evaluated["error"].mean())

    if "ape" in evaluated.columns:
        nonzero_ape = evaluated["ape"].dropna()
        metrics["mape"] = float(nonzero_ape.mean()) if not nonzero_ape.empty else None

    return metrics


def latest_prediction_table(pred_df: pd.DataFrame, limit: int = 24) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    latest_df = filter_to_latest_run(pred_df)
    latest_df = deduplicate_forecast_time(latest_df)
    latest_df = latest_df.sort_values("forecast_time").tail(limit)

    cols = [
        col for col in [
            "forecast_time",
            "y_pred",
            "lag_24",
            "prediction_type",
            "model_version",
            "run_id",
            "created_at",
        ]
        if col in latest_df.columns
    ]
    return latest_df[cols].copy()


def build_main_chart(chart_df: pd.DataFrame, show_baseline: bool = True) -> go.Figure:
    fig = go.Figure()

    if chart_df.empty:
        return fig

    past_df = chart_df[chart_df["actual_available"]].copy() if "actual_available" in chart_df.columns else pd.DataFrame()
    future_df = chart_df[~chart_df["actual_available"]].copy() if "actual_available" in chart_df.columns else pd.DataFrame()

    if not past_df.empty and "y_true" in past_df.columns:
        fig.add_trace(
            go.Scatter(
                x=past_df["forecast_time"],
                y=past_df["y_true"],
                mode="lines",
                name="Actual",
                line=dict(width=3),
                hovertemplate="Time: %{x}<br>Actual: %{y:,.2f}<extra></extra>",
            )
        )

    if not past_df.empty and "y_pred" in past_df.columns:
        fig.add_trace(
            go.Scatter(
                x=past_df["forecast_time"],
                y=past_df["y_pred"],
                mode="lines",
                name="Prediction (Past)",
                line=dict(width=2),
                hovertemplate="Time: %{x}<br>Prediction: %{y:,.2f}<extra></extra>",
            )
        )

    if not future_df.empty and "y_pred" in future_df.columns:
        fig.add_trace(
            go.Scatter(
                x=future_df["forecast_time"],
                y=future_df["y_pred"],
                mode="lines",
                name="Prediction (Future)",
                line=dict(width=3, dash="dash"),
                hovertemplate="Time: %{x}<br>Future Prediction: %{y:,.2f}<extra></extra>",
            )
        )

    if show_baseline and "lag_24" in chart_df.columns:
        baseline_df = chart_df.dropna(subset=["lag_24"]).copy()
        if not baseline_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=baseline_df["forecast_time"],
                    y=baseline_df["lag_24"],
                    mode="lines",
                    name="Baseline (lag_24)",
                    line=dict(width=1, dash="dot"),
                    hovertemplate="Time: %{x}<br>lag_24: %{y:,.2f}<extra></extra>",
                )
            )

    if not future_df.empty:
        future_start = future_df["forecast_time"].min()
        future_end = future_df["forecast_time"].max()

        fig.add_vrect(
            x0=future_start,
            x1=future_end,
            opacity=0.08,
            line_width=0,
            annotation_text="Future forecast window",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Actual vs Prediction vs Future Forecast",
        height=520,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(title_text="Forecast Time")
    fig.update_yaxes(title_text="PTF")
    return fig


def build_error_hour_chart(error_view: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if error_view.empty or "forecast_time" not in error_view.columns or "abs_error" not in error_view.columns:
        return fig

    hourly_error = (
        error_view.groupby(error_view["forecast_time"].dt.hour)["abs_error"]
        .mean()
        .reset_index(name="mean_abs_error")
        .rename(columns={"forecast_time": "hour"})
    )
    hourly_error = hourly_error.rename(columns={hourly_error.columns[0]: "hour"})

    fig.add_trace(
        go.Bar(
            x=hourly_error["hour"],
            y=hourly_error["mean_abs_error"],
            name="Mean Absolute Error",
            hovertemplate="Hour: %{x}<br>MAE: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Mean Absolute Error by Forecast Hour",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(title_text="Hour of Day")
    fig.update_yaxes(title_text="Mean Absolute Error")
    return fig


def build_daily_mae_chart(error_view: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if error_view.empty or "forecast_time" not in error_view.columns or "abs_error" not in error_view.columns:
        return fig

    daily_mae = (
        error_view.groupby(error_view["forecast_time"].dt.date)["abs_error"]
        .mean()
        .reset_index(name="daily_mae")
    )
    daily_mae = daily_mae.rename(columns={daily_mae.columns[0]: "date"})

    fig.add_trace(
        go.Scatter(
            x=daily_mae["date"],
            y=daily_mae["daily_mae"],
            mode="lines+markers",
            name="Daily MAE",
            hovertemplate="Date: %{x}<br>Daily MAE: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Daily MAE",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="MAE")
    return fig


def bias_label(bias: float | None) -> str:
    if bias is None:
        return "-"
    if bias > 0:
        return "Underprediction bias"
    if bias < 0:
        return "Overprediction bias"
    return "Neutral"


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

    model_versions = sorted(eval_df["model_version"].dropna().unique().tolist()) if "model_version" in eval_df.columns else []
    prediction_types = sorted(eval_df["prediction_type"].dropna().unique().tolist()) if "prediction_type" in eval_df.columns else []

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

    only_actuals = st.checkbox("Only evaluated rows", value=False)
    latest_run_only = st.checkbox("Use only latest run", value=True)
    show_baseline = st.checkbox("Show lag_24 baseline", value=True)

    min_time = eval_df["forecast_time"].min() if "forecast_time" in eval_df.columns and not eval_df.empty else None
    max_time = eval_df["forecast_time"].max() if "forecast_time" in eval_df.columns and not eval_df.empty else None

    default_date_value = None
    if pd.notna(min_time) and pd.notna(max_time):
        default_date_value = (min_time.date(), max_time.date())

    date_range = st.date_input(
        "Forecast date range",
        value=default_date_value,
    )

    chart_window = st.selectbox(
        "Chart window",
        options=[24, 72, 168, 336, 720],
        index=2,
        format_func=lambda x: f"Last {x} hours",
    )

filtered_df = eval_df.copy()

if selected_models and "model_version" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["model_version"].isin(selected_models)].copy()

if selected_types and "prediction_type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["prediction_type"].isin(selected_types)].copy()

filtered_df = apply_date_filter(filtered_df, date_range)

if latest_run_only:
    filtered_df = filter_to_latest_run(filtered_df)

filtered_df = deduplicate_forecast_time(filtered_df)

if only_actuals and "actual_available" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["actual_available"]].copy()

filtered_df = filtered_df.sort_values("forecast_time").reset_index(drop=True)

metrics = compute_metrics(filtered_df)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows", f"{metrics['rows']:,}")
c2.metric("Rows with actual", f"{metrics['rows_with_actual']:,}")
c3.metric("Future rows", f"{metrics['future_rows']:,}")
c4.metric("MAE", "-" if metrics["mae"] is None else f"{metrics['mae']:.2f}")
c5.metric("RMSE", "-" if metrics["rmse"] is None else f"{metrics['rmse']:.2f}")
c6.metric("Bias", "-" if metrics["bias"] is None else f"{metrics['bias']:.2f}")

bias_text = bias_label(metrics["bias"])
st.caption(f"Bias interpretation: {bias_text}")

if metrics["mape"] is not None:
    st.caption(
        f"MAPE: {metrics['mape']:.2f}% — note that this metric can be unstable for zero or near-zero electricity prices."
    )

chart_df = filtered_df.tail(chart_window).copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "Forecast Overview",
    "Error Analysis",
    "Latest Forecast",
    "Data Preview",
])

with tab1:
    st.subheader("Forecast Overview")

    if chart_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig = build_main_chart(chart_df, show_baseline=show_baseline)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Recent rows in chart window**")
            preview_cols = [
                col for col in [
                    "forecast_time",
                    "y_pred",
                    "y_true",
                    "lag_24",
                    "actual_available",
                    "error",
                    "abs_error",
                    "prediction_type",
                    "model_version",
                    "created_at",
                    "evaluation_created_at",
                ]
                if col in chart_df.columns
            ]
            st.dataframe(chart_df[preview_cols].tail(50), use_container_width=True)

        with col_b:
            latest_future = chart_df[~chart_df["actual_available"]].copy() if "actual_available" in chart_df.columns else pd.DataFrame()
            st.markdown("**Upcoming forecast horizon**")
            if latest_future.empty:
                st.info("No future-only forecast rows in the current chart window.")
            else:
                future_cols = [
                    col for col in [
                        "forecast_time",
                        "y_pred",
                        "lag_24",
                        "prediction_type",
                        "model_version",
                    ]
                    if col in latest_future.columns
                ]
                st.dataframe(latest_future[future_cols].head(48), use_container_width=True)

with tab2:
    st.subheader("Error Analysis")

    if filtered_df.empty:
        st.warning("No error data available for the selected filters.")
    else:
        error_view = filtered_df.copy()
        if "actual_available" in error_view.columns:
            error_view = error_view[error_view["actual_available"]].copy()

        if error_view.empty:
            st.info("There are no rows with actual values in the selected range.")
        else:
            fig_hour = build_error_hour_chart(error_view)
            st.plotly_chart(fig_hour, use_container_width=True)

            fig_day = build_daily_mae_chart(error_view)
            st.plotly_chart(fig_day, use_container_width=True)

            if "abs_error" in error_view.columns:
                worst_rows = error_view.sort_values("abs_error", ascending=False).head(20)
            else:
                worst_rows = error_view.head(20)

            worst_cols = [
                col for col in [
                    "forecast_time",
                    "y_pred",
                    "y_true",
                    "lag_24",
                    "error",
                    "abs_error",
                    "prediction_type",
                    "model_version",
                    "created_at",
                    "evaluation_created_at",
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

        if "forecast_time" in latest_df.columns and "y_pred" in latest_df.columns:
            latest_fig = go.Figure()

            latest_fig.add_trace(
                go.Scatter(
                    x=latest_df["forecast_time"],
                    y=latest_df["y_pred"],
                    mode="lines+markers",
                    name="Latest Prediction",
                    hovertemplate="Time: %{x}<br>Prediction: %{y:,.2f}<extra></extra>",
                )
            )

            if show_baseline and "lag_24" in latest_df.columns:
                baseline_df = latest_df.dropna(subset=["lag_24"]).copy()
                if not baseline_df.empty:
                    latest_fig.add_trace(
                        go.Scatter(
                            x=baseline_df["forecast_time"],
                            y=baseline_df["lag_24"],
                            mode="lines",
                            name="lag_24 baseline",
                            line=dict(dash="dot"),
                            hovertemplate="Time: %{x}<br>lag_24: %{y:,.2f}<extra></extra>",
                        )
                    )

            latest_fig.update_layout(
                title="Latest Prediction Curve",
                height=420,
                hovermode="x unified",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            latest_fig.update_xaxes(title_text="Forecast Time")
            latest_fig.update_yaxes(title_text="PTF")
            st.plotly_chart(latest_fig, use_container_width=True)

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