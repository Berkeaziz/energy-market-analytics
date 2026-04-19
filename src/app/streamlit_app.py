from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PTF Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  :root {
    --bg:#111827; --surface:#1f2937; --surface-2:#374151; --border:#4b5563;
    --accent:#38bdf8; --accent-2:#818cf8; --success:#34d399; --warning:#f59e0b; --danger:#f87171;
    --text:#f3f4f6; --muted:#9ca3af; --mono:'IBM Plex Mono', monospace; --sans:'Inter', sans-serif;
  }
  html, body, [class*="css"] { font-family: var(--sans) !important; background-color: var(--bg) !important; color: var(--text) !important; }
  body { background: var(--bg) !important; }
  [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #1f2937 0%, #111827 100%) !important; }
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(56,189,248,0.06), transparent 24%),
      radial-gradient(circle at top right, rgba(129,140,248,0.05), transparent 22%),
      linear-gradient(180deg, #1f2937 0%, #111827 100%) !important;
  }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #111827; }
  ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 8px; }
  section[data-testid="stSidebar"] { background: linear-gradient(180deg, #374151 0%, #1f2937 100%) !important; border-right: 1px solid #4b5563 !important; }
  section[data-testid="stSidebar"] * { color: #f3f4f6 !important; }
  section[data-testid="stSidebar"] .stMarkdown h1, section[data-testid="stSidebar"] .stMarkdown h2, section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important; font-family: var(--mono) !important; letter-spacing: .04em; font-size: .76rem !important; text-transform: uppercase;
  }
  section[data-testid="stSidebar"] hr { border-color: rgba(156,163,175,0.18) !important; }
  section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] label p, section[data-testid="stSidebar"] .stCheckbox label, section[data-testid="stSidebar"] .stRadio label {
    color: #f9fafb !important; opacity: 1 !important; font-weight: 500 !important;
  }
  label[data-testid="stWidgetLabel"] {
    color: #e5e7eb !important; font-family: var(--mono) !important; font-size: .72rem !important; letter-spacing: .06em !important; text-transform: uppercase !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #111827 !important; border: 1px solid #4b5563 !important; border-radius: 12px !important; box-shadow: none !important; min-height: 44px !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #111827 !important; color: #f9fafb !important; }
  section[data-testid="stSidebar"] [data-baseweb="select"] input { color: #f9fafb !important; }
  section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder { color: #9ca3af !important; }
  section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #2563eb !important; border: 1px solid #3b82f6 !important; color: #ffffff !important; border-radius: 999px !important; font-family: var(--mono) !important; font-size: .70rem !important; padding: 2px 6px !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="tag"] span, section[data-testid="stSidebar"] [data-baseweb="tag"] div { color: #ffffff !important; }
  section[data-testid="stSidebar"] [data-baseweb="tag"] svg { fill: #ffffff !important; color: #ffffff !important; }
  section[data-testid="stSidebar"] svg { color: #cbd5e1 !important; fill: #cbd5e1 !important; }
  section[data-testid="stSidebar"] .stCheckbox label span { color: #e5e7eb !important; }
  .ptf-header { display: flex; align-items: center; gap: .8rem; padding: 1.25rem 0 .75rem; border-bottom: 1px solid rgba(159,176,199,0.18); margin-bottom: 1rem; }
  .ptf-header h1 { font-family: var(--sans) !important; font-size: 1.9rem !important; font-weight: 700 !important; color: var(--text) !important; margin: 0 !important; }
  .ptf-header .badge {
    font-family: var(--mono); font-size: .66rem; padding: 4px 10px; border: 1px solid rgba(56,189,248,0.28); background: rgba(56,189,248,0.08); border-radius: 999px; color: var(--accent); text-transform: uppercase; letter-spacing: .08em;
  }
  .ptf-caption { font-size: .84rem; color: var(--muted); margin-bottom: 1.2rem; }
  [data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(31,41,55,0.96), rgba(17,24,39,0.96)) !important; border: 1px solid rgba(159,176,199,0.16) !important; border-radius: 16px !important; padding: 1rem !important; box-shadow: 0 8px 24px rgba(0,0,0,0.16);
  }
  [data-testid="stMetricLabel"] {
    font-family: var(--mono) !important; font-size: .66rem !important; text-transform: uppercase !important; letter-spacing: .08em !important; color: var(--muted) !important;
  }
  [data-testid="stMetricValue"] {
    font-family: var(--sans) !important; font-size: 1.45rem !important; font-weight: 700 !important; color: var(--text) !important;
  }
  .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid rgba(159,176,199,0.16) !important; gap: .35rem !important; }
  section[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    background-color: #1f2937 !important; color: #f3f4f6 !important; border-radius: 10px !important; border: 1px solid #4b5563 !important; padding: 8px !important;
  }
  section[data-testid="stSidebar"] [data-testid="stDateInput"] input::placeholder { color: #9ca3af !important; }
  .stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important; font-size: .72rem !important; text-transform: uppercase !important; letter-spacing: .08em !important; color: var(--muted) !important; border: 1px solid transparent !important; border-radius: 10px 10px 0 0 !important; padding: .7rem 1rem !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--accent) !important; background: rgba(56,189,248,0.07) !important; border-color: rgba(56,189,248,0.18) !important; border-bottom-color: transparent !important;
  }
  .info-banner, .warn-banner, .error-banner, .success-banner {
    border-radius: 14px; padding: .8rem 1rem; font-size: .84rem; margin-bottom: 1rem; border: 1px solid transparent;
  }
  .info-banner { background: rgba(56,189,248,0.08); border-color: rgba(56,189,248,0.18); color: var(--text); }
  .success-banner { background: rgba(52,211,153,0.08); border-color: rgba(52,211,153,0.18); color: var(--text); }
  .warn-banner { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.18); color: var(--text); }
  .error-banner { background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.18); color: var(--text); }
  .section-header {
    font-family: var(--mono) !important; font-size: .72rem !important; text-transform: uppercase !important; letter-spacing: .12em !important; color: var(--muted) !important; margin-bottom: .75rem !important; padding-bottom: .45rem !important; border-bottom: 1px solid rgba(159,176,199,0.12) !important;
  }
  .stDataFrame { border: 1px solid rgba(159,176,199,0.14) !important; border-radius: 16px !important; overflow: hidden !important; }
  .stDataFrame [data-testid="stDataFrameResizable"] { background: rgba(31,41,55,0.94) !important; }
  .stJson { background: rgba(31,41,55,0.92) !important; border: 1px solid rgba(159,176,199,0.14) !important; border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()

EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "ptf" / "ptf_prediction_evaluation.parquet"
SUMMARY_PATH = PROJECT_ROOT / "data" / "evaluation" / "ptf" / "ptf_prediction_evaluation_summary.json"
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "ptf" / "ptf_predictions_history.parquet"
DECISION_PATH = PROJECT_ROOT / "data" / "decision" / "ptf" / "ptf_decision_signals.parquet"
DECISION_SUMMARY_PATH = PROJECT_ROOT / "data" / "decision" / "ptf" / "ptf_decision_summary.json"
SIM_PATH = PROJECT_ROOT / "data" / "decision" / "ptf" / "ptf_strategy_simulation.parquet"
SIM_SUMMARY_PATH = PROJECT_ROOT / "data" / "decision" / "ptf" / "ptf_strategy_simulation_summary.json"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#9fb0c7"),
    xaxis=dict(gridcolor="#334155", zeroline=False, showline=True, linecolor="#334155"),
    yaxis=dict(gridcolor="#334155", zeroline=False, showline=False),
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
    strategy="#22c55e",
    risk="#ef4444",
)

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
    if "actual_available" in df.columns:
        df["actual_available"] = df["actual_available"].fillna(False).astype(bool)
    if "forecast_time" in df.columns:
        df = df.sort_values("forecast_time").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_decision_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "forecast_time" in df.columns:
        df["forecast_time"] = to_naive_datetime(df["forecast_time"])
        df = df.sort_values("forecast_time").reset_index(drop=True)
    numeric_cols = [
        "y_pred", "pred_mean_horizon", "pred_std_horizon", "pred_vs_mean_ratio",
        "pred_zscore_horizon", "pred_diff_1", "pred_abs_diff_1", "local_volatility", "risk_score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_simulation_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "forecast_time" in df.columns:
        df["forecast_time"] = to_naive_datetime(df["forecast_time"])
        df = df.sort_values("forecast_time").reset_index(drop=True)
    numeric_cols = [
        "y_pred", "ptf", "smf", "baseline_bid_mwh", "strategy_multiplier", "strategy_bid_mwh",
        "actual_generation_mwh", "baseline_imbalance_mwh", "strategy_imbalance_mwh",
        "baseline_dayahead_revenue", "strategy_dayahead_revenue", "baseline_imbalance_cashflow",
        "strategy_imbalance_cashflow", "baseline_total_revenue", "strategy_total_revenue",
        "delta_dayahead_revenue", "delta_imbalance_cashflow", "delta_total_revenue",
        "baseline_abs_imbalance_mwh", "strategy_abs_imbalance_mwh", "delta_abs_imbalance_mwh"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def safe_json_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_for_display(df: pd.DataFrame, formats: dict[str, str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if not formats:
        return out
    for col, fmt in formats.items():
        if col not in out.columns:
            continue
        mask = out[col].notna()
        out[col] = out[col].astype(object)
        out.loc[mask, col] = out.loc[mask, col].map(lambda x: fmt.format(x))
    return out

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
    _, today_start, today_end = get_today_bounds()
    if "created_at" in latest_df.columns and latest_df["created_at"].notna().any():
        status["latest_created_at"] = latest_df["created_at"].max()
    if "forecast_time" in latest_df.columns and latest_df["forecast_time"].notna().any():
        status["latest_forecast_time"] = latest_df["forecast_time"].max()
        today_mask = ((latest_df["forecast_time"] >= today_start) & (latest_df["forecast_time"] <= today_end))
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
    out = out[(out["forecast_time"] >= now) & (out["forecast_time"] < end_time)].copy()
    return out.sort_values("forecast_time").reset_index(drop=True)

def compute_spread_stats(sim_df: pd.DataFrame) -> dict:
    base = {"mean_spread": None, "median_spread": None, "std_spread": None, "min_spread": None, "max_spread": None}
    if sim_df.empty or not {"ptf", "smf"}.issubset(sim_df.columns):
        return base
    spread = (pd.to_numeric(sim_df["smf"], errors="coerce") - pd.to_numeric(sim_df["ptf"], errors="coerce")).dropna()
    if spread.empty:
        return base
    base["mean_spread"] = float(spread.mean())
    base["median_spread"] = float(spread.median())
    base["std_spread"] = float(spread.std())
    base["min_spread"] = float(spread.min())
    base["max_spread"] = float(spread.max())
    return base

def build_overview_df(
    eval_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    selected_models: list[str],
    selected_types: list[str],
    date_range,
    latest_run_only: bool,
) -> pd.DataFrame:
    pred_view = pred_df.copy()

    if selected_models and "model_version" in pred_view.columns:
        pred_view = pred_view[pred_view["model_version"].isin(selected_models)].copy()
    if selected_types and "prediction_type" in pred_view.columns:
        pred_view = pred_view[pred_view["prediction_type"].isin(selected_types)].copy()

    pred_view = apply_date_filter(pred_view, date_range)
    if latest_run_only:
        pred_view = filter_to_latest_run(pred_view)
    pred_view = deduplicate_forecast_time(pred_view)

    eval_view = eval_df.copy()
    if selected_models and "model_version" in eval_view.columns:
        eval_view = eval_view[eval_view["model_version"].isin(selected_models)].copy()
    if selected_types and "prediction_type" in eval_view.columns:
        eval_view = eval_view[eval_view["prediction_type"].isin(selected_types)].copy()

    eval_view = apply_date_filter(eval_view, date_range)
    if latest_run_only:
        eval_view = filter_to_latest_run(eval_view)
    eval_view = deduplicate_forecast_time(eval_view)

    merge_cols = [c for c in [
        "forecast_time", "y_true", "error", "abs_error", "squared_error", "ape",
        "actual_available", "evaluation_created_at"
    ] if c in eval_view.columns]

    if pred_view.empty and not eval_view.empty:
        out = eval_view.copy()
        if "y_pred" not in out.columns:
            out["y_pred"] = np.nan
        if "actual_available" not in out.columns:
            out["actual_available"] = out["y_true"].notna()
        return out.sort_values("forecast_time").reset_index(drop=True)

    if pred_view.empty:
        return pd.DataFrame()

    if merge_cols:
        out = pred_view.merge(eval_view[merge_cols], on="forecast_time", how="left")
    else:
        out = pred_view.copy()

    if "actual_available" not in out.columns:
        out["actual_available"] = False
    out["actual_available"] = out["actual_available"].fillna(False).astype(bool)

    return out.sort_values("forecast_time").reset_index(drop=True)

def build_main_chart(chart_df: pd.DataFrame, show_baseline: bool = True) -> go.Figure:
    fig = go.Figure()
    if chart_df.empty:
        return fig

    has_avail = "actual_available" in chart_df.columns
    past_df = chart_df[chart_df["actual_available"]].copy() if has_avail else pd.DataFrame()
    future_df = chart_df[~chart_df["actual_available"]].copy() if has_avail else chart_df.copy()

    def smart_mode(df_: pd.DataFrame) -> str:
        return "lines+markers" if len(df_) <= 1 else "lines"

    if not past_df.empty and "y_true" in past_df.columns:
        fig.add_trace(go.Scatter(
            x=past_df["forecast_time"],
            y=past_df["y_true"],
            mode=smart_mode(past_df),
            name="Actual",
            line=dict(color=COLORS["actual"], width=2.5),
            marker=dict(size=8, color=COLORS["actual"]),
            hovertemplate="<b>Actual</b>: %{y:,.1f}<extra></extra>"
        ))

    if not past_df.empty and "y_pred" in past_df.columns:
        fig.add_trace(go.Scatter(
            x=past_df["forecast_time"],
            y=past_df["y_pred"],
            mode=smart_mode(past_df),
            name="Prediction",
            line=dict(color=COLORS["pred_past"], width=1.8),
            marker=dict(size=7, color=COLORS["pred_past"]),
            hovertemplate="<b>Prediction</b>: %{y:,.1f}<extra></extra>"
        ))

    if not future_df.empty and "y_pred" in future_df.columns:
        fig.add_trace(go.Scatter(
            x=future_df["forecast_time"],
            y=future_df["y_pred"],
            mode=smart_mode(future_df),
            name="Future Forecast",
            line=dict(color=COLORS["pred_future"], width=2, dash="dash"),
            marker=dict(size=8, color=COLORS["pred_future"]),
            hovertemplate="<b>Future</b>: %{y:,.1f}<extra></extra>"
        ))

    if show_baseline and "lag_24" in chart_df.columns:
        bl = chart_df.dropna(subset=["lag_24"]).copy()
        if not bl.empty:
            fig.add_trace(go.Scatter(
                x=bl["forecast_time"],
                y=bl["lag_24"],
                mode="lines+markers" if len(bl) <= 1 else "lines",
                name="Lag-24 Baseline",
                line=dict(color=COLORS["baseline"], width=1.2, dash="dot"),
                marker=dict(size=6, color=COLORS["baseline"]),
                opacity=0.7,
                hovertemplate="<b>Lag-24</b>: %{y:,.1f}<extra></extra>"
            ))

    if not future_df.empty:
        x0 = future_df["forecast_time"].min()
        x1 = future_df["forecast_time"].max()
        if pd.notna(x0) and pd.notna(x1):
            if x0 == x1:
                x1 = x1 + pd.Timedelta(minutes=1)
            fig.add_vrect(
                x0=x0,
                x1=x1,
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
    hourly = error_view.groupby(error_view["forecast_time"].dt.hour)["abs_error"].mean().reset_index()
    hourly.columns = ["hour", "mae"]
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["mae"], name="Hourly MAE",
        marker=dict(color=hourly["mae"], colorscale=[[0, "#818cf8"], [1, "#f87171"]], showscale=False),
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
    daily = error_view.groupby(error_view["forecast_time"].dt.date)["abs_error"].mean().reset_index()
    daily.columns = ["date", "daily_mae"]
    daily["rolling_7d"] = daily["daily_mae"].rolling(7, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["daily_mae"], mode="lines", name="Daily MAE",
                             line=dict(color=COLORS["mae"], width=1.2), opacity=0.6,
                             hovertemplate="<b>%{x}</b><br>MAE: %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling_7d"], mode="lines", name="7-Day Rolling",
                             line=dict(color=COLORS["actual"], width=2),
                             hovertemplate="<b>%{x}</b><br>7d Avg: %{y:,.2f}<extra></extra>"))
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
    fig.add_trace(go.Histogram(x=errors, nbinsx=60, name="Error distribution",
                               marker=dict(color="#818cf8", opacity=0.8),
                               hovertemplate="Error: %{x:,.1f}<br>Count: %{y}<extra></extra>"))
    fig.add_vline(x=0, line_color="#38bdf8", line_width=1.5, line_dash="dash")
    fig.add_vline(x=float(errors.mean()), line_color="#f59e0b", line_width=1.5,
                  annotation_text=f"μ={errors.mean():.1f}", annotation_font=dict(size=9, color="#f59e0b"))
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=320, title=dict(text="Error Distribution", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Forecast Error", title_font=dict(size=10))
    fig.update_yaxes(title_text="Count", title_font=dict(size=10))
    return fig

def build_decision_chart(decision_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if decision_df.empty:
        return fig
    fig.add_trace(go.Scatter(x=decision_df["forecast_time"], y=decision_df["y_pred"], mode="lines+markers", name="Forecast Price",
                             line=dict(color=COLORS["pred_future"], width=2.4), marker=dict(size=6),
                             hovertemplate="<b>%{x}</b><br>y_pred: %{y:,.2f}<extra></extra>"))
    if "risk_score" in decision_df.columns:
        risky = decision_df[decision_df["risk_score"] >= 1.5].copy()
        if not risky.empty:
            fig.add_trace(go.Scatter(x=risky["forecast_time"], y=risky["y_pred"], mode="markers", name="Risky Hours",
                                     marker=dict(color=COLORS["risk"], size=11, symbol="diamond"),
                                     hovertemplate="<b>%{x}</b><br>risk_score: %{customdata:.2f}<extra></extra>",
                                     customdata=risky["risk_score"]))
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=430, title=dict(text="Decision Signals over Forecast Horizon", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Forecast Time", title_font=dict(size=10))
    fig.update_yaxes(title_text="Forecast PTF", title_font=dict(size=10))
    return fig

def build_strategy_revenue_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sim_df.empty or "delta_total_revenue" not in sim_df.columns:
        return fig
    delta = pd.to_numeric(sim_df["delta_total_revenue"], errors="coerce")
    fig.add_trace(go.Bar(x=sim_df["forecast_time"], y=delta, name="Delta Revenue",
                         hovertemplate="<b>%{x}</b><br>Δ Revenue: %{y:,.2f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=380, title=dict(text="Strategy Revenue Improvement vs Baseline", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Forecast Time", title_font=dict(size=10))
    fig.update_yaxes(title_text="Delta Revenue", title_font=dict(size=10))
    return fig

def build_strategy_multiplier_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sim_df.empty or "strategy_multiplier" not in sim_df.columns:
        return fig
    mult = pd.to_numeric(sim_df["strategy_multiplier"], errors="coerce")
    fig.add_trace(go.Scatter(x=sim_df["forecast_time"], y=mult, mode="lines+markers", name="Strategy Multiplier",
                             line=dict(color=COLORS["pred_future"], width=2.2), marker=dict(size=4),
                             hovertemplate="<b>%{x}</b><br>Multiplier: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#94a3b8")
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=320, title=dict(text="Strategy Multiplier Over Time", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Forecast Time", title_font=dict(size=10))
    fig.update_yaxes(title_text="Multiplier", title_font=dict(size=10))
    return fig

def build_bid_difference_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    required = {"baseline_bid_mwh", "strategy_bid_mwh"}
    if sim_df.empty or not required.issubset(sim_df.columns):
        return fig
    baseline_bid = pd.to_numeric(sim_df["baseline_bid_mwh"], errors="coerce")
    strategy_bid = pd.to_numeric(sim_df["strategy_bid_mwh"], errors="coerce")
    bid_diff = strategy_bid - baseline_bid
    fig.add_trace(go.Bar(x=sim_df["forecast_time"], y=bid_diff, name="Bid Difference",
                         hovertemplate="<b>%{x}</b><br>Strategy - Baseline: %{y:,.2f} MWh<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=320, title=dict(text="Strategy Bid Difference vs Baseline", font=dict(size=13, color="#e2e8f0")))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Forecast Time", title_font=dict(size=10))
    fig.update_yaxes(title_text="Δ Bid (MWh)", title_font=dict(size=10))
    return fig

def build_smf_ptf_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sim_df.empty:
        return fig
    if "ptf" in sim_df.columns:
        fig.add_trace(go.Scatter(x=sim_df["forecast_time"], y=sim_df["ptf"], mode="lines", name="PTF", line=dict(width=2)))
    if "smf" in sim_df.columns:
        fig.add_trace(go.Scatter(x=sim_df["forecast_time"], y=sim_df["smf"], mode="lines", name="SMF", line=dict(width=2, dash="dot")))
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=350, title="PTF vs SMF")
    fig.update_layout(**layout)
    return fig

def build_spread_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sim_df.empty:
        return fig
    if {"ptf", "smf"}.issubset(sim_df.columns):
        spread = pd.to_numeric(sim_df["smf"], errors="coerce") - pd.to_numeric(sim_df["ptf"], errors="coerce")
        fig.add_trace(go.Bar(x=sim_df["forecast_time"], y=spread, name="SMF - PTF",
                             hovertemplate="<b>%{x}</b><br>Spread: %{y:,.2f}<extra></extra>"))
        fig.add_hline(y=0, line_dash="dash")
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=350, title="SMF - PTF Spread")
    fig.update_layout(**layout)
    return fig

def build_spread_hist(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sim_df.empty:
        return fig
    if {"ptf", "smf"}.issubset(sim_df.columns):
        spread = (pd.to_numeric(sim_df["smf"], errors="coerce") - pd.to_numeric(sim_df["ptf"], errors="coerce")).dropna()
        fig.add_trace(go.Histogram(x=spread, nbinsx=50, name="Spread Distribution"))
        fig.add_vline(x=0, line_dash="dash")
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(height=300, title="Spread Distribution")
    fig.update_layout(**layout)
    return fig

try:
    eval_df = load_eval_data(EVAL_PATH)
    pred_df = load_prediction_data(PRED_PATH)
    summary = safe_json_summary(SUMMARY_PATH)
    decision_df = load_decision_data(DECISION_PATH)
    decision_summary = safe_json_summary(DECISION_SUMMARY_PATH)
    sim_df = load_simulation_data(SIM_PATH)
    sim_summary = safe_json_summary(SIM_SUMMARY_PATH)
except Exception as e:
    st.markdown(f'<div class="error-banner">⚠ {e}</div>', unsafe_allow_html=True)
    st.stop()

with st.sidebar:
    st.markdown("### PTF Dashboard")
    st.markdown('<p class="ptf-caption">EPİAŞ PTF Forecast Monitor</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### Filters")
    model_versions = sorted(
        list(set(eval_df["model_version"].dropna().unique().tolist()) | set(pred_df["model_version"].dropna().unique().tolist()))
    ) if ("model_version" in eval_df.columns or "model_version" in pred_df.columns) else []

    prediction_types = sorted(
        list(set(eval_df["prediction_type"].dropna().unique().tolist()) | set(pred_df["prediction_type"].dropna().unique().tolist()))
    ) if ("prediction_type" in eval_df.columns or "prediction_type" in pred_df.columns) else []

    selected_models = st.multiselect("Model version", options=model_versions, default=model_versions[-1:] if model_versions else [])
    selected_types = st.multiselect("Prediction type", options=prediction_types, default=prediction_types)

    st.divider()
    st.markdown("#### Options")
    only_actuals = st.checkbox("Only evaluated rows", value=False)
    latest_run_only = st.checkbox("Use only latest run", value=False)
    show_baseline = st.checkbox("Show lag-24 baseline", value=True)

    st.divider()
    st.markdown("#### Date Range")
    all_times = []
    if "forecast_time" in eval_df.columns and not eval_df.empty:
        all_times.append(eval_df["forecast_time"].dropna())
    if "forecast_time" in pred_df.columns and not pred_df.empty:
        all_times.append(pred_df["forecast_time"].dropna())
    if "forecast_time" in decision_df.columns and not decision_df.empty:
        all_times.append(decision_df["forecast_time"].dropna())
    if "forecast_time" in sim_df.columns and not sim_df.empty:
        all_times.append(sim_df["forecast_time"].dropna())

    _, today_start, today_end = get_today_bounds()
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
    chart_window = st.select_slider("Hours to display", options=[24, 48, 72, 168, 336, 720], value=168, format_func=lambda x: f"{x}h")

overview_df = build_overview_df(
    eval_df=eval_df,
    pred_df=pred_df,
    selected_models=selected_models,
    selected_types=selected_types,
    date_range=date_range,
    latest_run_only=latest_run_only,
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

filtered_decision_df = apply_date_filter(decision_df, date_range)
filtered_decision_df = deduplicate_forecast_time(filtered_decision_df)
filtered_decision_df = filtered_decision_df.sort_values("forecast_time").reset_index(drop=True)

filtered_sim_df = apply_date_filter(sim_df, date_range)
filtered_sim_df = deduplicate_forecast_time(filtered_sim_df)
filtered_sim_df = filtered_sim_df.sort_values("forecast_time").reset_index(drop=True)

metrics = compute_metrics(overview_df if not only_actuals else filtered_df)
prediction_status = get_prediction_status(pred_df)
_, today_start, _ = get_today_bounds()

st.markdown("""
<div class="ptf-header">
  <h1> PTF Forecast Dashboard</h1>
  <span class="badge">EPİAŞ</span>
  <span class="badge">Live</span>
</div>
<p class="ptf-caption">Turkey Electricity Balancing Market · Piyasa Takas Fiyatı prediction, decision & strategy monitor</p>
""", unsafe_allow_html=True)

if pred_df.empty:
    st.markdown('<div class="error-banner">Prediction history dosyası boş. Dashboard yeni forecast gösteremez.</div>', unsafe_allow_html=True)
else:
    latest_created_txt = prediction_status["latest_created_at"].strftime("%Y-%m-%d %H:%M") if prediction_status["latest_created_at"] is not None else "—"
    latest_forecast_txt = prediction_status["latest_forecast_time"].strftime("%Y-%m-%d %H:%M") if prediction_status["latest_forecast_time"] is not None else "—"
    if prediction_status["has_today_forecast"]:
        st.markdown(f'<div class="success-banner">✅ Güncel tahmin var · today_rows={prediction_status["today_rows"]} · latest_created_at={latest_created_txt} · latest_forecast_time={latest_forecast_txt}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-banner">⚠ Bugün ({today_start.strftime("%d-%m-%Y")}) için forecast görünmüyor. latest_created_at={latest_created_txt} · latest_forecast_time={latest_forecast_txt}</div>', unsafe_allow_html=True)

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

if metrics["bias"] is not None:
    bias_txt, _ = bias_label(metrics["bias"])
    icon = "↑" if metrics["bias"] > 0 else "↓" if metrics["bias"] < 0 else "="
    st.markdown(f'<div class="info-banner">{icon} Bias: {bias_txt}</div>', unsafe_allow_html=True)
if metrics["mape"] is not None:
    mape_color = "warn-banner" if metrics["mape"] > 20 else "info-banner"
    st.markdown(f'<div class="{mape_color}">MAPE: {metrics["mape"]:.2f}% — note: can be unstable for near-zero prices</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Forecast Overview", "Error Analysis", "Latest Forecast", "Decision Signals", "Strategy Simulation", "Data Preview",
])

with tab1:
    chart_df = overview_df.copy()
    if only_actuals and "actual_available" in chart_df.columns:
        chart_df = chart_df[chart_df["actual_available"]].copy()
    chart_df = chart_df.tail(chart_window).copy()

    if chart_df.empty:
        st.markdown('<div class="warn-banner">⚠ No data available for the selected filters.</div>', unsafe_allow_html=True)
    else:
        st.plotly_chart(build_main_chart(chart_df, show_baseline=show_baseline), use_container_width=True)
        col_a, col_b = st.columns([3, 2], gap="medium")
        with col_a:
            st.markdown('<p class="section-header">Recent rows in chart window</p>', unsafe_allow_html=True)
            preview_cols = [c for c in [
                "forecast_time", "y_pred", "y_true", "lag_24", "actual_available", "error",
                "abs_error", "prediction_type", "model_version", "created_at", "evaluation_created_at"
            ] if c in chart_df.columns]
            st.dataframe(
                format_for_display(
                    chart_df[preview_cols].tail(50),
                    {"y_pred": "{:,.2f}", "y_true": "{:,.2f}", "error": "{:,.2f}", "abs_error": "{:,.2f}", "lag_24": "{:,.2f}"}
                ),
                use_container_width=True,
                height=380
            )
        with col_b:
            st.markdown('<p class="section-header">Upcoming forecast horizon</p>', unsafe_allow_html=True)
            future_rows = chart_df[~chart_df["actual_available"]].copy() if "actual_available" in chart_df.columns else pd.DataFrame()
            if future_rows.empty:
                st.markdown('<div class="info-banner">ℹ No future-only rows in current chart window.</div>', unsafe_allow_html=True)
            else:
                future_cols = [c for c in ["forecast_time", "y_pred", "lag_24", "prediction_type", "model_version"] if c in future_rows.columns]
                st.dataframe(
                    format_for_display(future_rows[future_cols].head(48), {"y_pred": "{:,.2f}", "lag_24": "{:,.2f}"}),
                    use_container_width=True,
                    height=380
                )

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
            worst_rows = error_view.sort_values("abs_error", ascending=False).head(20) if "abs_error" in error_view.columns else error_view.head(20)
            worst_cols = [c for c in ["forecast_time", "y_pred", "y_true", "lag_24", "error", "abs_error", "prediction_type", "model_version", "created_at", "evaluation_created_at"] if c in worst_rows.columns]
            st.markdown('<p class="section-header">Top 20 worst forecast errors</p>', unsafe_allow_html=True)
            st.dataframe(format_for_display(worst_rows[worst_cols], {"y_pred": "{:,.2f}", "y_true": "{:,.2f}", "error": "{:,.2f}", "abs_error": "{:,.2f}", "lag_24": "{:,.2f}"}), use_container_width=True)

with tab3:
    upcoming_df = get_upcoming_24h_predictions(pred_df)
    if upcoming_df.empty:
        st.markdown('<div class="warn-banner">⚠ Önümüzdeki 24 saat için prediction history içinde forecast satırı bulunamadı.</div>', unsafe_allow_html=True)
        latest_df = latest_prediction_table(pred_df, limit=24)
        if not latest_df.empty:
            st.markdown('<div class="info-banner">ℹ Referans olarak son mevcut tahminler gösteriliyor.</div>', unsafe_allow_html=True)
            st.dataframe(format_for_display(latest_df, {"y_pred": "{:,.2f}", "lag_24": "{:,.2f}"}), use_container_width=True, height=420)
    else:
        c1, c2 = st.columns([2, 3], gap="medium")
        with c1:
            st.markdown('<p class="section-header">Upcoming 24H forecast</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="success-banner">✅ Önümüzdeki 24 saat için {len(upcoming_df)} adet tahmin bulundu.</div>', unsafe_allow_html=True)
            fmt = {}
            if "y_pred" in upcoming_df.columns:
                fmt["y_pred"] = "{:,.2f}"
            if "lag_24" in upcoming_df.columns:
                fmt["lag_24"] = "{:,.2f}"
            show_cols = [c for c in ["forecast_time", "y_pred", "lag_24", "prediction_type", "model_version", "created_at"] if c in upcoming_df.columns]
            st.dataframe(format_for_display(upcoming_df[show_cols], fmt), use_container_width=True, height=460)
        with c2:
            forecast_fig = go.Figure()
            forecast_fig.add_trace(go.Scatter(x=upcoming_df["forecast_time"], y=upcoming_df["y_pred"], mode="lines+markers", name="Prediction",
                                              line=dict(color=COLORS["pred_future"], width=2.8), marker=dict(size=6, color=COLORS["pred_future"]),
                                              hovertemplate="<b>%{x}</b><br>Prediction: %{y:,.2f}<extra></extra>"))
            if show_baseline and "lag_24" in upcoming_df.columns:
                bl = upcoming_df.dropna(subset=["lag_24"])
                if not bl.empty:
                    forecast_fig.add_trace(go.Scatter(x=bl["forecast_time"], y=bl["lag_24"], mode="lines", name="Lag-24 Baseline",
                                                      line=dict(color=COLORS["baseline"], width=1.6, dash="dot"), opacity=0.85,
                                                      hovertemplate="<b>%{x}</b><br>Lag-24: %{y:,.2f}<extra></extra>"))
            now_line = pd.Timestamp.now().to_pydatetime()
            forecast_fig.add_vline(x=now_line, line_width=1, line_dash="dash", line_color=COLORS["actual"])
            forecast_fig.add_annotation(x=now_line, y=1, yref="paper", text="NOW", showarrow=False, yshift=10, font=dict(size=10, color=COLORS["actual"]))
            layout = dict(**PLOTLY_LAYOUT)
            layout.update(height=460, title=dict(text="Next 24 Hours Forecast", font=dict(size=13, color="#e2e8f0")))
            forecast_fig.update_layout(**layout)
            forecast_fig.update_xaxes(title_text="Forecast Time", title_font=dict(size=10))
            forecast_fig.update_yaxes(title_text="PTF (TL/MWh)", title_font=dict(size=10))
            st.plotly_chart(forecast_fig, use_container_width=True)

with tab4:
    if filtered_decision_df.empty:
        st.markdown('<div class="warn-banner">⚠ Decision signals dosyası boş ya da seçilen tarihte veri yok.</div>', unsafe_allow_html=True)
    else:
        d1, d2, d3, d4 = st.columns(4)
        signal_counts = decision_summary.get("signal_counts", {})
        risk_counts = decision_summary.get("risk_counts", {})
        with d1:
            st.metric("Decision Rows", f"{len(filtered_decision_df):,}")
        with d2:
            st.metric("High Value Hours", signal_counts.get("HIGH_VALUE_HOUR", 0))
        with d3:
            st.metric("High Risk Hours", risk_counts.get("high_risk", 0))
        with d4:
            avg_pred = decision_summary.get("avg_pred")
            st.metric("Avg Pred", "—" if avg_pred is None else f"{avg_pred:,.2f}")
        st.plotly_chart(build_decision_chart(filtered_decision_df.tail(chart_window)), use_container_width=True)
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            st.markdown('<p class="section-header">Decision signal table</p>', unsafe_allow_html=True)
            show_cols = [c for c in ["forecast_time", "y_pred", "price_regime", "risk_score", "risk_label", "decision_signal", "decision_note", "pred_zscore_horizon", "pred_abs_diff_1", "local_volatility"] if c in filtered_decision_df.columns]
            st.dataframe(format_for_display(filtered_decision_df[show_cols], {"y_pred": "{:,.2f}", "risk_score": "{:.2f}", "pred_zscore_horizon": "{:.2f}", "pred_abs_diff_1": "{:,.2f}", "local_volatility": "{:,.2f}"}), use_container_width=True, height=430)
        with c2:
            st.markdown('<p class="section-header">Decision summary JSON</p>', unsafe_allow_html=True)
            st.json(decision_summary if decision_summary else {"info": "decision summary not found"})

with tab5:
    if filtered_sim_df.empty:
        st.markdown('<div class="warn-banner">⚠ Strategy simulation dosyası boş ya da seçilen tarihte veri yok.</div>', unsafe_allow_html=True)
    else:
        strategy_view = filtered_sim_df.tail(chart_window).copy()
        spread_stats = compute_spread_stats(strategy_view)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            val = sim_summary.get("total_baseline_revenue")
            st.metric("Baseline Revenue", "—" if val is None else f"{val:,.0f}")
        with s2:
            val = sim_summary.get("total_strategy_revenue")
            st.metric("Strategy Revenue", "—" if val is None else f"{val:,.0f}")
        with s3:
            val = sim_summary.get("total_gain")
            st.metric("Total Gain", "—" if val is None else f"{val:,.0f}")
        with s4:
            val = sim_summary.get("imbalance_reduction_mwh")
            st.metric("Imbalance Reduction", "—" if val is None else f"{val:,.2f}")

        mean_spread_txt = "—" if spread_stats["mean_spread"] is None else f"{spread_stats['mean_spread']:,.2f}"
        std_spread_txt = "—" if spread_stats["std_spread"] is None else f"{spread_stats['std_spread']:,.2f}"
        min_spread_txt = "—" if spread_stats["min_spread"] is None else f"{spread_stats['min_spread']:,.2f}"
        max_spread_txt = "—" if spread_stats["max_spread"] is None else f"{spread_stats['max_spread']:,.2f}"
        st.markdown(
            "<div class='info-banner'>"
            f"Mean spread (SMF - PTF): {mean_spread_txt} · "
            f"Std spread: {std_spread_txt} · "
            f"Min: {min_spread_txt} · "
            f"Max: {max_spread_txt}"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Market Analysis (SMF vs PTF)")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.plotly_chart(build_smf_ptf_chart(strategy_view), use_container_width=True)
        with c2:
            st.plotly_chart(build_spread_chart(strategy_view), use_container_width=True)
        st.plotly_chart(build_spread_hist(strategy_view), use_container_width=True)

        st.markdown("### Strategy Impact")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.plotly_chart(build_strategy_revenue_chart(strategy_view), use_container_width=True)
        with c2:
            st.plotly_chart(build_strategy_multiplier_chart(strategy_view), use_container_width=True)
        st.plotly_chart(build_bid_difference_chart(strategy_view), use_container_width=True)

        st.markdown('<p class="section-header">Strategy simulation table</p>', unsafe_allow_html=True)
        show_cols = [c for c in [
            "forecast_time", "decision_signal", "risk_label", "y_pred", "ptf", "smf", "baseline_bid_mwh",
            "strategy_multiplier", "strategy_bid_mwh", "actual_generation_mwh", "baseline_total_revenue",
            "strategy_total_revenue", "delta_total_revenue", "baseline_abs_imbalance_mwh",
            "strategy_abs_imbalance_mwh", "delta_abs_imbalance_mwh"
        ] if c in strategy_view.columns]
        st.dataframe(format_for_display(strategy_view[show_cols], {
            "y_pred": "{:,.2f}", "ptf": "{:,.2f}", "smf": "{:,.2f}", "baseline_bid_mwh": "{:,.2f}",
            "strategy_multiplier": "{:.2f}", "strategy_bid_mwh": "{:,.2f}", "actual_generation_mwh": "{:,.2f}",
            "baseline_total_revenue": "{:,.2f}", "strategy_total_revenue": "{:,.2f}",
            "delta_total_revenue": "{:,.2f}", "baseline_abs_imbalance_mwh": "{:,.2f}",
            "strategy_abs_imbalance_mwh": "{:,.2f}", "delta_abs_imbalance_mwh": "{:,.2f}",
        }), use_container_width=True, height=460)
        st.markdown('<p class="section-header">Strategy summary JSON</p>', unsafe_allow_html=True)
        st.json(sim_summary if sim_summary else {"info": "simulation summary not found"})

with tab6:
    c1, c2, c3 = st.columns([2, 1, 1], gap="medium")
    with c1:
        st.markdown('<p class="section-header">Evaluation summary JSON</p>', unsafe_allow_html=True)
        st.json(summary if summary else {"info": "summary.json not found"})
    with c2:
        st.markdown('<p class="section-header">Filtered dataframe shape</p>', unsafe_allow_html=True)
        rows, cols = overview_df.shape
        st.markdown(
            f"""
            <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem;">
              <div style="font-family:var(--mono);font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem;">Rows</div>
              <div style="font-family:var(--mono);font-size:1.6rem;font-weight:600;color:var(--accent)">{rows:,}</div>
              <div style="font-family:var(--mono);font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin:.5rem 0 .25rem;">Columns</div>
              <div style="font-family:var(--mono);font-size:1.6rem;font-weight:600;color:var(--accent-2)">{cols}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown('<p class="section-header">Available columns</p>', unsafe_allow_html=True)
        cols_html = "".join([
            f'<div style="font-family:var(--mono);font-size:.7rem;padding:.2rem .5rem;margin:.15rem 0;background:rgba(56,189,248,.07);border-left:2px solid var(--accent);border-radius:6px;color:var(--text)">{c}</div>'
            for c in overview_df.columns.tolist()
        ])
        st.markdown(f'<div style="max-height:280px;overflow-y:auto">{cols_html}</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header" style="margin-top:1.5rem">Overview rows (last 200)</p>', unsafe_allow_html=True)
    st.dataframe(overview_df.tail(200), use_container_width=True)