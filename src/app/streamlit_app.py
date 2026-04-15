from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PTF Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  :root {
    --bg:        #111827;
    --surface:   #1f2937;
    --surface-2: #374151;
    --border:    #4b5563;

    --accent:    #38bdf8;
    --accent-2:  #818cf8;
    --success:   #34d399;
    --warning:   #f59e0b;
    --danger:    #f87171;

    --text:      #f3f4f6;
    --muted:     #9ca3af;

    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Inter', sans-serif;
  }

  html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  body {
    background: var(--bg) !important;
  }

  [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #1f2937 0%, #111827 100%) !important;
  }

  .stApp {
    background:
      radial-gradient(circle at top left, rgba(56,189,248,0.06), transparent 24%),
      radial-gradient(circle at top right, rgba(129,140,248,0.05), transparent 22%),
      linear-gradient(180deg, #1f2937 0%, #111827 100%) !important;
  }

  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #111827; }
  ::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 8px;
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #374151 0%, #1f2937 100%) !important;
    border-right: 1px solid #4b5563 !important;
  }

  section[data-testid="stSidebar"] * {
    color: #f3f4f6 !important;
  }

  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    letter-spacing: .04em;
    font-size: .76rem !important;
    text-transform: uppercase;
  }

  section[data-testid="stSidebar"] hr {
    border-color: rgba(156,163,175,0.18) !important;
  }

  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] label p,
  section[data-testid="stSidebar"] .stCheckbox label,
  section[data-testid="stSidebar"] .stRadio label {
    color: #f9fafb !important;
    opacity: 1 !important;
    font-weight: 500 !important;
  }

  label[data-testid="stWidgetLabel"] {
    color: #e5e7eb !important;
    font-family: var(--mono) !important;
    font-size: .72rem !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #111827 !important;
    border: 1px solid #4b5563 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    min-height: 44px !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #111827 !important;
    color: #f9fafb !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #f9fafb !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
    color: #9ca3af !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #2563eb !important;
    border: 1px solid #3b82f6 !important;
    color: #ffffff !important;
    border-radius: 999px !important;
    font-family: var(--mono) !important;
    font-size: .70rem !important;
    padding: 2px 6px !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="tag"] span,
  section[data-testid="stSidebar"] [data-baseweb="tag"] div {
    color: #ffffff !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="tag"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
  }

  section[data-testid="stSidebar"] svg {
    color: #cbd5e1 !important;
    fill: #cbd5e1 !important;
  }

  section[data-testid="stSidebar"] .stCheckbox label span {
    color: #e5e7eb !important;
  }

  .ptf-header {
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: 1.25rem 0 .75rem;
    border-bottom: 1px solid rgba(159,176,199,0.18);
    margin-bottom: 1rem;
  }

  .ptf-header h1 {
    font-family: var(--sans) !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin: 0 !important;
  }

  .ptf-header .badge {
    font-family: var(--mono);
    font-size: .66rem;
    padding: 4px 10px;
    border: 1px solid rgba(56,189,248,0.28);
    background: rgba(56,189,248,0.08);
    border-radius: 999px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: .08em;
  }

  .ptf-caption {
    font-size: .84rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
  }

  [data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(31,41,55,0.96), rgba(17,24,39,0.96)) !important;
    border: 1px solid rgba(159,176,199,0.16) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.16);
  }

  [data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: .66rem !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    color: var(--muted) !important;
  }

  [data-testid="stMetricValue"] {
    font-family: var(--sans) !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid rgba(159,176,199,0.16) !important;
    gap: .35rem !important;
  }

	/* DATE PICKER FULL */
	section[data-testid="stSidebar"] [data-testid="stDateInput"] input {
		background-color: #1f2937 !important;
		color: #f3f4f6 !important;
		border-radius: 10px !important;
		border: 1px solid #4b5563 !important;
		padding: 8px !important;
	}

	/* iç yazı */
	section[data-testid="stSidebar"] [data-testid="stDateInput"] input {
		color: #f9fafb !important;
	}

	/* placeholder */
	section[data-testid="stSidebar"] [data-testid="stDateInput"] input::placeholder {
		color: #9ca3af !important;
	}
  .stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: .72rem !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    color: var(--muted) !important;
    border: 1px solid transparent !important;
    border-radius: 10px 10px 0 0 !important;
    padding: .7rem 1rem !important;
  }

  .stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: rgba(56,189,248,0.07) !important;
    border-color: rgba(56,189,248,0.18) !important;
    border-bottom-color: transparent !important;
  }

  .info-banner, .warn-banner, .error-banner, .success-banner {
    border-radius: 14px;
    padding: .8rem 1rem;
    font-size: .84rem;
    margin-bottom: 1rem;
    border: 1px solid transparent;
  }

  .info-banner {
    background: rgba(56,189,248,0.08);
    border-color: rgba(56,189,248,0.18);
    color: var(--text);
  }

  .success-banner {
    background: rgba(52,211,153,0.08);
    border-color: rgba(52,211,153,0.18);
    color: var(--text);
  }

  .warn-banner {
    background: rgba(245,158,11,0.08);
    border-color: rgba(245,158,11,0.18);
    color: var(--text);
  }

  .error-banner {
    background: rgba(248,113,113,0.08);
    border-color: rgba(248,113,113,0.18);
    color: var(--text);
  }

  .section-header {
    font-family: var(--mono) !important;
    font-size: .72rem !important;
    text-transform: uppercase !important;
    letter-spacing: .12em !important;
    color: var(--muted) !important;
    margin-bottom: .75rem !important;
    padding-bottom: .45rem !important;
    border-bottom: 1px solid rgba(159,176,199,0.12) !important;
  }

  .stDataFrame {
    border: 1px solid rgba(159,176,199,0.14) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
  }

  .stDataFrame [data-testid="stDataFrameResizable"] {
    background: rgba(31,41,55,0.94) !important;
  }

  .stCheckbox label,
  label[data-testid="stWidgetLabel"] {
    color: var(--text) !important;
  }

  .stJson {
    background: rgba(31,41,55,0.92) !important;
    border: 1px solid rgba(159,176,199,0.14) !important;
    border-radius: 16px !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
EVAL_PATH    = PROJECT_ROOT / "data" / "evaluation" / "ptf" / "ptf_prediction_evaluation.parquet"
SUMMARY_PATH = PROJECT_ROOT / "data" / "evaluation" / "ptf" / "ptf_prediction_evaluation_summary.json"
PRED_PATH    = PROJECT_ROOT / "data" / "predictions" / "ptf" / "ptf_predictions_history.parquet"

# ─── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#9fb0c7"),
    xaxis=dict(
        gridcolor="#334155",
        zeroline=False,
        showline=True,
        linecolor="#334155",
    ),
    yaxis=dict(
        gridcolor="#334155",
        zeroline=False,
        showline=False,
    ),
    legend=dict(
        bgcolor="rgba(30,41,59,.85)",
        bordercolor="#334155",
        borderwidth=1,
        font=dict(size=10),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=10, r=10, t=54, b=10),
    hovermode="x unified",
)

COLORS = dict(
    actual="#38bdf8",
    pred_past="#818cf8",
    pred_future="#34d399",
    baseline="#f59e0b",
    mae="#f87171",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_naive_datetime(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series, errors="coerce")
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_localize(None)
    return series


@st.cache_data(show_spinner=False)
def load_eval_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    df = pd.read_parquet(path)
    for col in ["feature_time", "forecast_time", "created_at", "evaluation_created_at"]:
        if col in df.columns:
            df[col] = to_naive_datetime(df[col])
    for col in ["y_pred", "y_true", "error", "abs_error", "squared_error", "ape", "lag_24", "lag_168"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "actual_available" in df.columns:
        df["actual_available"] = df["actual_available"].fillna(False).astype(bool)
    if "forecast_time" in df.columns:
        df = df.sort_values("forecast_time").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_prediction_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    for col in ["feature_time", "forecast_time", "created_at"]:
        if col in df.columns:
            df[col] = to_naive_datetime(df[col])
    for col in ["y_pred", "lag_24", "lag_168"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "forecast_time" in df.columns:
        df = df.sort_values("forecast_time").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def safe_json_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_to_latest_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ["created_at", "evaluation_created_at"]:
        if col in out.columns and out[col].notna().any():
            latest_ts = out[col].dropna().max()
            return out[out[col] == latest_ts].copy()
    return out


def deduplicate_forecast_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "forecast_time" not in df.columns:
        return df.copy()
    out = df.copy()
    sort_cols = ["forecast_time"]
    for c in ["created_at", "evaluation_created_at"]:
        if c in out.columns:
            sort_cols.append(c)
            break
    out = out.sort_values(sort_cols)
    out = out.drop_duplicates(subset=["forecast_time"], keep="last")
    return out.sort_values("forecast_time").reset_index(drop=True)


def apply_date_filter(df: pd.DataFrame, date_range) -> pd.DataFrame:
    if df.empty or "forecast_time" not in df.columns:
        return df.copy()
    if not (isinstance(date_range, tuple) and len(date_range) == 2):
        return df.copy()
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["forecast_time"] >= start) & (df["forecast_time"] <= end)].copy()


def compute_metrics(df: pd.DataFrame) -> dict:
    base = dict(rows=0, rows_with_actual=0, mae=None, rmse=None, mape=None, bias=None, future_rows=0)
    if df.empty:
        return base
    evaluated = df[df["actual_available"]].copy() if "actual_available" in df.columns else pd.DataFrame()
    base["rows"] = int(len(df))
    base["rows_with_actual"] = int(len(evaluated))
    base["future_rows"] = int((~df["actual_available"]).sum()) if "actual_available" in df.columns else 0
    if evaluated.empty:
        return base
    if "abs_error" in evaluated.columns:
        base["mae"] = float(evaluated["abs_error"].mean())
    if "squared_error" in evaluated.columns:
        base["rmse"] = float(np.sqrt(evaluated["squared_error"].mean()))
    if "error" in evaluated.columns:
        base["bias"] = float(evaluated["error"].mean())
    if "ape" in evaluated.columns:
        v = evaluated["ape"].dropna()
        if not v.empty:
            base["mape"] = float(v.mean())
    return base


def latest_prediction_table(pred_df: pd.DataFrame, limit: int = 24) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    latest_df = filter_to_latest_run(pred_df)
    latest_df = deduplicate_forecast_time(latest_df)
    latest_df = latest_df.sort_values("forecast_time").tail(limit)
    cols = [c for c in ["forecast_time", "y_pred", "lag_24", "prediction_type", "model_version", "run_id", "created_at"] if c in latest_df.columns]
    return latest_df[cols].copy()


def bias_label(bias) -> tuple[str, str]:
    if bias is None:
        return "—", "muted"
    if bias > 0:
        return f"+{bias:.2f} · Underprediction", "warning"
    if bias < 0:
        return f"{bias:.2f} · Overprediction", "danger"
    return "Neutral", "accent3"


def get_today_bounds():
    now = pd.Timestamp.now().floor("min")
    today_start = now.normalize()
    today_end = today_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return now, today_start, today_end


def get_prediction_status(pred_df: pd.DataFrame) -> dict:
    status = {
        "latest_created_at": None,
        "latest_forecast_time": None,
        "has_today_forecast": False,
        "today_rows": 0,
        "is_stale": True,
    }

    if pred_df.empty:
        return status

    latest_df = filter_to_latest_run(pred_df)
    latest_df = deduplicate_forecast_time(latest_df)

    if latest_df.empty:
        return status

    now, today_start, today_end = get_today_bounds()

    if "created_at" in latest_df.columns and latest_df["created_at"].notna().any():
        status["latest_created_at"] = latest_df["created_at"].max()

    if "forecast_time" in latest_df.columns and latest_df["forecast_time"].notna().any():
        status["latest_forecast_time"] = latest_df["forecast_time"].max()
        today_mask = (
            (latest_df["forecast_time"] >= today_start) &
            (latest_df["forecast_time"] <= today_end)
        )
        status["today_rows"] = int(today_mask.sum())
        status["has_today_forecast"] = bool(today_mask.any())
        status["is_stale"] = status["latest_forecast_time"] < today_start

    return status
def get_upcoming_24h_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or "forecast_time" not in pred_df.columns:
        return pd.DataFrame()

    out = pred_df.copy()

    sort_cols = ["forecast_time"]
    if "created_at" in out.columns:
        sort_cols.append("created_at")

    out = out.sort_values(sort_cols)
    out = out.drop_duplicates(subset=["forecast_time"], keep="last")
    out = out.sort_values("forecast_time").reset_index(drop=True)

    now = pd.Timestamp.now().floor("h")
    end_time = now + pd.Timedelta(hours=24)

    out = out[
        (out["forecast_time"] >= now) &
        (out["forecast_time"] < end_time)
    ].copy()

    return out.sort_values("forecast_time").reset_index(drop=True)

# ─── Chart builders ───────────────────────────────────────────────────────────
def build_main_chart(chart_df: pd.DataFrame, show_baseline: bool = True) -> go.Figure:
    fig = go.Figure()
    if chart_df.empty:
        return fig

    has_avail = "actual_available" in chart_df.columns
    past_df = chart_df[chart_df["actual_available"]].copy() if has_avail else pd.DataFrame()
    future_df = chart_df[~chart_df["actual_available"]].copy() if has_avail else pd.DataFrame()

    if not past_df.empty and "y_true" in past_df.columns:
        fig.add_trace(go.Scatter(
            x=past_df["forecast_time"], y=past_df["y_true"],
            mode="lines", name="Actual",
            line=dict(color=COLORS["actual"], width=2.5),
            hovertemplate="<b>Actual</b>: %{y:,.1f}<extra></extra>",
        ))

    if not past_df.empty and "y_pred" in past_df.columns:
        fig.add_trace(go.Scatter(
            x=past_df["forecast_time"], y=past_df["y_pred"],
            mode="lines", name="Prediction",
            line=dict(color=COLORS["pred_past"], width=1.8),
            hovertemplate="<b>Prediction</b>: %{y:,.1f}<extra></extra>",
        ))

    if not future_df.empty and "y_pred" in future_df.columns:
        fig.add_trace(go.Scatter(
            x=future_df["forecast_time"], y=future_df["y_pred"],
            mode="lines", name="Future Forecast",
            line=dict(color=COLORS["pred_future"], width=2, dash="dash"),
            hovertemplate="<b>Future</b>: %{y:,.1f}<extra></extra>",
        ))

    if show_baseline and "lag_24" in chart_df.columns:
        bl = chart_df.dropna(subset=["lag_24"]).copy()
        if not bl.empty:
            fig.add_trace(go.Scatter(
                x=bl["forecast_time"], y=bl["lag_24"],
                mode="lines", name="Lag-24 Baseline",
                line=dict(color=COLORS["baseline"], width=1.2, dash="dot"),
                opacity=0.7,
                hovertemplate="<b>Lag-24</b>: %{y:,.1f}<extra></extra>",
            ))

    if not future_df.empty:
        fig.add_vrect(
            x0=future_df["forecast_time"].min(),
            x1=future_df["forecast_time"].max(),
            fillcolor="rgba(52,211,153,.05)",
            line_width=0,
            annotation_text="FORECAST WINDOW",
            annotation_font=dict(size=9, color="#34d399", family="IBM Plex Mono"),
            annotation_position="top left",
        )

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=480, title=dict(text="Actual vs Prediction vs Future Forecast", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Forecast Time", title_font=dict(size=10))
    fig.update_yaxes(title_text="PTF (TL/MWh)", title_font=dict(size=10))
    return fig


def build_error_hour_chart(error_view: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if error_view.empty or "forecast_time" not in error_view.columns or "abs_error" not in error_view.columns:
        return fig

    hourly = (
        error_view.groupby(error_view["forecast_time"].dt.hour)["abs_error"]
        .mean().reset_index()
    )
    hourly.columns = ["hour", "mae"]

    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["mae"],
        name="Hourly MAE",
        marker=dict(
            color=hourly["mae"],
            colorscale=[[0, "#818cf8"], [1, "#f87171"]],
            showscale=False,
        ),
        hovertemplate="<b>Hour %{x}</b><br>MAE: %{y:,.2f}<extra></extra>",
    ))
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=320, title=dict(text="Mean Absolute Error by Hour of Day", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Hour of Day", dtick=2, title_font=dict(size=10))
    fig.update_yaxes(title_text="MAE", title_font=dict(size=10))
    return fig


def build_daily_mae_chart(error_view: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if error_view.empty or "forecast_time" not in error_view.columns or "abs_error" not in error_view.columns:
        return fig

    daily = (
        error_view.groupby(error_view["forecast_time"].dt.date)["abs_error"]
        .mean().reset_index()
    )
    daily.columns = ["date", "daily_mae"]
    daily["rolling_7d"] = daily["daily_mae"].rolling(7, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["daily_mae"],
        mode="lines",
        name="Daily MAE",
        line=dict(color=COLORS["mae"], width=1.2),
        opacity=0.6,
        hovertemplate="<b>%{x}</b><br>MAE: %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling_7d"],
        mode="lines",
        name="7-Day Rolling",
        line=dict(color=COLORS["actual"], width=2),
        hovertemplate="<b>%{x}</b><br>7d Avg: %{y:,.2f}<extra></extra>",
    ))
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=320, title=dict(text="Daily MAE with 7-Day Rolling Average", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Date", title_font=dict(size=10))
    fig.update_yaxes(title_text="MAE", title_font=dict(size=10))
    return fig


def build_error_dist_chart(error_view: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if error_view.empty or "error" not in error_view.columns:
        return fig

    errors = error_view["error"].dropna()

    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=60,
        name="Error distribution",
        marker=dict(color="#818cf8", opacity=0.8),
        hovertemplate="Error: %{x:,.1f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#38bdf8", line_width=1.5, line_dash="dash")
    fig.add_vline(
        x=float(errors.mean()),
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text=f"μ={errors.mean():.1f}",
        annotation_font=dict(size=9, color="#f59e0b"),
    )

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=320, title=dict(text="Error Distribution", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Forecast Error", title_font=dict(size=10))
    fig.update_yaxes(title_text="Count", title_font=dict(size=10))
    return fig


# ─── Load data ────────────────────────────────────────────────────────────────
try:
    eval_df = load_eval_data(EVAL_PATH)
    pred_df = load_prediction_data(PRED_PATH)
    summary = safe_json_summary(SUMMARY_PATH)
except Exception as e:
    st.markdown(f'<div class="error-banner">⚠ {e}</div>', unsafe_allow_html=True)
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ PTF Dashboard")
    st.markdown('<p class="ptf-caption">EPİAŞ PTF Forecast Monitor</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### Filters")

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

    st.divider()
    st.markdown("#### Options")

    only_actuals = st.checkbox("Only evaluated rows", value=False)
    latest_run_only = st.checkbox("Use only latest run", value=True)
    show_baseline = st.checkbox("Show lag-24 baseline", value=True)

    st.divider()
    st.markdown("#### Date Range")

    all_times = []

    if "forecast_time" in eval_df.columns and not eval_df.empty:
        all_times.append(eval_df["forecast_time"].dropna())

    if "forecast_time" in pred_df.columns and not pred_df.empty:
        all_times.append(pred_df["forecast_time"].dropna())

    now, today_start, today_end = get_today_bounds()

    if all_times:
        combined_times = pd.concat(all_times, ignore_index=True)
        min_time = combined_times.min()
        max_time = combined_times.max()
    else:
        min_time, max_time = None, None

    if pd.notna(min_time) and pd.notna(max_time):
        date_end = max(max_time.date(), today_end.date())
        default_date = (min_time.date(), date_end)
    else:
        default_date = (today_start.date(), today_end.date())

    date_range = st.date_input("Forecast date range", value=default_date)

    st.divider()
    st.markdown("#### Chart Window")

    chart_window = st.select_slider(
        "Hours to display",
        options=[24, 48, 72, 168, 336, 720],
        value=168,
        format_func=lambda x: f"{x}h",
    )

# ─── Filter logic ─────────────────────────────────────────────────────────────
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
prediction_status = get_prediction_status(pred_df)
now, today_start, today_end = get_today_bounds()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ptf-header">
  <h1>⚡ PTF Forecast Dashboard</h1>
  <span class="badge">EPİAŞ</span>
  <span class="badge">Live</span>
</div>
<p class="ptf-caption">Turkey Electricity Balancing Market · Piyasa Takas Fiyatı prediction &amp; evaluation monitor</p>
""", unsafe_allow_html=True)

if pred_df.empty:
    st.markdown(
        '<div class="error-banner">Prediction history dosyası boş. Dashboard yeni forecast gösteremez.</div>',
        unsafe_allow_html=True,
    )
else:
    latest_created_txt = (
        prediction_status["latest_created_at"].strftime("%Y-%m-%d %H:%M")
        if prediction_status["latest_created_at"] is not None else "—"
    )
    latest_forecast_txt = (
        prediction_status["latest_forecast_time"].strftime("%Y-%m-%d %H:%M")
        if prediction_status["latest_forecast_time"] is not None else "—"
    )

    if prediction_status["has_today_forecast"]:
        st.markdown(
            f'<div class="success-banner">✅ Güncel tahmin var · today_rows={prediction_status["today_rows"]} · latest_created_at={latest_created_txt} · latest_forecast_time={latest_forecast_txt}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="warn-banner">⚠ Bugün ({today_start.strftime("%d-%m-%Y")}) için forecast görünmüyor. latest_created_at={latest_created_txt} · latest_forecast_time={latest_forecast_txt}</div>',
            unsafe_allow_html=True,
        )

# ─── Metric cards ─────────────────────────────────────────────────────────────
bias_txt, bias_color = bias_label(metrics["bias"])

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Rows", f"{metrics['rows']:,}")
with col2:
    st.metric("Evaluated Rows", f"{metrics['rows_with_actual']:,}")
with col3:
    st.metric("Future Rows", f"{metrics['future_rows']:,}")
with col4:
    st.metric("MAE", "—" if metrics["mae"] is None else f"{metrics['mae']:.2f}")
with col5:
    st.metric("RMSE", "—" if metrics["rmse"] is None else f"{metrics['rmse']:.2f}")
with col6:
    st.metric("Bias", "—" if metrics["bias"] is None else f"{metrics['bias']:.2f}")

# ── Status banners ────────────────────────────────────────────────────────────
if metrics["bias"] is not None:
    icon = "↑" if metrics["bias"] > 0 else "↓" if metrics["bias"] < 0 else "="
    st.markdown(f'<div class="info-banner">{icon} Bias: {bias_txt}</div>', unsafe_allow_html=True)

if metrics["mape"] is not None:
    mape_color = "warn-banner" if metrics["mape"] > 20 else "info-banner"
    st.markdown(f'<div class="{mape_color}">MAPE: {metrics["mape"]:.2f}% — note: can be unstable for near-zero prices</div>', unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Forecast Overview",
    "🎯  Error Analysis",
    "🔭  Latest Forecast",
    "🗃  Data Preview",
])

# ══════════════════════ TAB 1: Forecast Overview ══════════════════════════════
with tab1:
    chart_df = filtered_df.tail(chart_window).copy()

    if chart_df.empty:
        st.markdown('<div class="warn-banner">⚠ No data available for the selected filters.</div>', unsafe_allow_html=True)
    else:
        fig = build_main_chart(chart_df, show_baseline=show_baseline)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns([3, 2], gap="medium")

        with col_a:
            st.markdown('<p class="section-header">Recent rows in chart window</p>', unsafe_allow_html=True)
            preview_cols = [c for c in [
                "forecast_time", "y_pred", "y_true", "lag_24",
                "actual_available", "error", "abs_error",
                "prediction_type", "model_version", "created_at", "evaluation_created_at",
            ] if c in chart_df.columns]
            st.dataframe(
                chart_df[preview_cols].tail(50).style.format({
                    "y_pred": "{:,.2f}", "y_true": "{:,.2f}",
                    "error": "{:,.2f}", "abs_error": "{:,.2f}",
                    "lag_24": "{:,.2f}",
                }),
                use_container_width=True,
                height=380,
            )

        with col_b:
            st.markdown('<p class="section-header">Upcoming forecast horizon</p>', unsafe_allow_html=True)
            if "actual_available" in chart_df.columns:
                future_rows = chart_df[~chart_df["actual_available"]].copy()
            else:
                future_rows = pd.DataFrame()

            if future_rows.empty:
                st.markdown('<div class="info-banner">ℹ No future-only rows in current chart window.</div>', unsafe_allow_html=True)
            else:
                future_cols = [c for c in [
                    "forecast_time", "y_pred", "lag_24", "prediction_type", "model_version",
                ] if c in future_rows.columns]
                st.dataframe(
                    future_rows[future_cols].head(48).style.format({
                        "y_pred": "{:,.2f}", "lag_24": "{:,.2f}",
                    }),
                    use_container_width=True,
                    height=380,
                )

# ══════════════════════ TAB 2: Error Analysis ═════════════════════════════════
with tab2:
    if filtered_df.empty:
        st.markdown('<div class="warn-banner">⚠ No error data for selected filters.</div>', unsafe_allow_html=True)
    else:
        error_view = filtered_df.copy()
        if "actual_available" in error_view.columns:
            error_view = error_view[error_view["actual_available"]].copy()

        if error_view.empty:
            st.markdown('<div class="info-banner">ℹ No rows with actual values in the selected range.</div>', unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.plotly_chart(build_error_hour_chart(error_view), use_container_width=True)
            with c2:
                st.plotly_chart(build_error_dist_chart(error_view), use_container_width=True)

            st.plotly_chart(build_daily_mae_chart(error_view), use_container_width=True)

            if "abs_error" in error_view.columns:
                worst_rows = error_view.sort_values("abs_error", ascending=False).head(20)
            else:
                worst_rows = error_view.head(20)

            worst_cols = [c for c in [
                "forecast_time", "y_pred", "y_true", "lag_24",
                "error", "abs_error", "prediction_type", "model_version",
                "created_at", "evaluation_created_at",
            ] if c in worst_rows.columns]

            st.markdown('<p class="section-header">Top 20 worst forecast errors</p>', unsafe_allow_html=True)
            st.dataframe(
                worst_rows[worst_cols].style.format({
                    "y_pred": "{:,.2f}", "y_true": "{:,.2f}",
                    "error": "{:,.2f}", "abs_error": "{:,.2f}",
                    "lag_24": "{:,.2f}",
                }).background_gradient(subset=["abs_error"] if "abs_error" in worst_cols else [], cmap="Reds"),
                use_container_width=True,
            )

# ══════════════════════ TAB 3: Latest Forecast ════════════════════════════════
with tab3:
    upcoming_df = get_upcoming_24h_predictions(pred_df)

    if upcoming_df.empty:
        st.markdown(
            '<div class="warn-banner">⚠ Önümüzdeki 24 saat için prediction history içinde forecast satırı bulunamadı.</div>',
            unsafe_allow_html=True,
        )

        latest_df = latest_prediction_table(pred_df, limit=24)

        if not latest_df.empty:
            st.markdown(
                '<div class="info-banner">ℹ Referans olarak son mevcut tahminler gösteriliyor.</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                latest_df.style.format({
                    "y_pred": "{:,.2f}",
                    "lag_24": "{:,.2f}",
                }),
                use_container_width=True,
                height=420,
            )
    else:
        c1, c2 = st.columns([2, 3], gap="medium")

        with c1:
            st.markdown('<p class="section-header">Upcoming 24H forecast</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="success-banner">✅ Önümüzdeki 24 saat için {len(upcoming_df)} adet tahmin bulundu.</div>',
                unsafe_allow_html=True,
            )

            fmt = {}
            if "y_pred" in upcoming_df.columns:
                fmt["y_pred"] = "{:,.2f}"
            if "lag_24" in upcoming_df.columns:
                fmt["lag_24"] = "{:,.2f}"

            show_cols = [
                c for c in [
                    "forecast_time",
                    "y_pred",
                    "lag_24",
                    "prediction_type",
                    "model_version",
                    "created_at",
                ] if c in upcoming_df.columns
            ]

            st.dataframe(
                upcoming_df[show_cols].style.format(fmt),
                use_container_width=True,
                height=460,
            )

        with c2:
            forecast_fig = go.Figure()

            forecast_fig.add_trace(go.Scatter(
                x=upcoming_df["forecast_time"],
                y=upcoming_df["y_pred"],
                mode="lines+markers",
                name="Prediction",
                line=dict(color=COLORS["pred_future"], width=2.8),
                marker=dict(size=6, color=COLORS["pred_future"]),
                hovertemplate="<b>%{x}</b><br>Prediction: %{y:,.2f}<extra></extra>",
            ))

            if show_baseline and "lag_24" in upcoming_df.columns:
                bl = upcoming_df.dropna(subset=["lag_24"])
                if not bl.empty:
                    forecast_fig.add_trace(go.Scatter(
                        x=bl["forecast_time"],
                        y=bl["lag_24"],
                        mode="lines",
                        name="Lag-24 Baseline",
                        line=dict(color=COLORS["baseline"], width=1.6, dash="dot"),
                        opacity=0.85,
                        hovertemplate="<b>%{x}</b><br>Lag-24: %{y:,.2f}<extra></extra>",
                    ))

            now_line = pd.Timestamp.now().to_pydatetime()

            forecast_fig.add_vline(
                x=now_line,
                line_width=1,
                line_dash="dash",
                line_color=COLORS["actual"],
            )

            forecast_fig.add_annotation(
                x=now_line,
                y=1,
                yref="paper",
                text="NOW",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color=COLORS["actual"]),
            )

            layout = dict(**PLOTLY_LAYOUT)
            layout.update(
                height=460,
                title=dict(
                    text="Next 24 Hours Forecast",
                    font=dict(size=13, color="#e2e8f0"),
                ),
            )
            forecast_fig.update_layout(**layout)
            forecast_fig.update_xaxes(title_text="Forecast Time", title_font=dict(size=10))
            forecast_fig.update_yaxes(title_text="PTF (TL/MWh)", title_font=dict(size=10))

            st.plotly_chart(forecast_fig, use_container_width=True)
# ══════════════════════ TAB 4: Data Preview ═══════════════════════════════════
with tab4:
    c1, c2, c3 = st.columns([2, 1, 1], gap="medium")

    with c1:
        st.markdown('<p class="section-header">Evaluation summary JSON</p>', unsafe_allow_html=True)
        st.json(summary if summary else {"info": "summary.json not found"})

    with c2:
        st.markdown('<p class="section-header">Filtered dataframe shape</p>', unsafe_allow_html=True)
        rows, cols = filtered_df.shape
        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem;">
          <div style="font-family:var(--mono);font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem;">Rows</div>
          <div style="font-family:var(--mono);font-size:1.6rem;font-weight:600;color:var(--accent)">{rows:,}</div>
          <div style="font-family:var(--mono);font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin:.5rem 0 .25rem;">Columns</div>
          <div style="font-family:var(--mono);font-size:1.6rem;font-weight:600;color:var(--accent-2)">{cols}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown('<p class="section-header">Available columns</p>', unsafe_allow_html=True)
        cols_html = "".join([
            f'<div style="font-family:var(--mono);font-size:.7rem;padding:.2rem .5rem;margin:.15rem 0;'
            f'background:rgba(56,189,248,.07);border-left:2px solid var(--accent);border-radius:6px;'
            f'color:var(--text)">{c}</div>'
            for c in filtered_df.columns.tolist()
        ])
        st.markdown(f'<div style="max-height:280px;overflow-y:auto">{cols_html}</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header" style="margin-top:1.5rem">Filtered evaluation rows (last 200)</p>', unsafe_allow_html=True)
    st.dataframe(filtered_df.tail(200), use_container_width=True)