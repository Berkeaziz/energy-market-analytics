from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow"
PYTHON_BIN = "python"
LOCAL_TZ = "Europe/Istanbul"

default_args = {
    "owner": "berke",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="inference_ptf_latest_pipeline",
    description=(
        "PTF latest inference pipeline with auxiliary forecasts: "
        "fetch/process raw data -> "
        "aux latest feature build -> aux latest predict -> "
        "fill missing externals in ptf latest features -> "
        "predict latest ptf -> decision -> simulate -> evaluate"
    ),
    start_date=pendulum.datetime(2026, 4, 1, tz=LOCAL_TZ),
    schedule="0 * * * *",
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    default_args=default_args,
    tags=[
        "ptf",
        "inference",
        "latest",
        "weather",
        "generation",
        "consumption",
        "smf",
        "auxiliary",
        "decision",
        "strategy",
    ],
    render_template_as_native_obj=True,
) as dag:

    # =========================================================
    # FETCH / PROCESS RAW DATA
    # =========================================================
    fetch_epias = BashOperator(
        task_id="fetch_epias",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/ingestion/fetch_epias_ptf.py
        """,
    )

    process_ptf = BashOperator(
        task_id="process_ptf",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/processing/process_ptf.py
        """,
    )

    fetch_weather = BashOperator(
        task_id="fetch_weather",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/ingestion/fetch_weather.py
        """,
    )

    process_weather = BashOperator(
        task_id="process_weather",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/processing/process_weather.py
        """,
    )

    fetch_generation = BashOperator(
        task_id="fetch_generation",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/ingestion/fetch_generation.py
        """,
    )

    process_generation = BashOperator(
        task_id="process_generation",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/processing/process_generation.py
        """,
    )

    fetch_consumption = BashOperator(
        task_id="fetch_consumption",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/ingestion/fetch_consumption.py
        """,
    )

    process_consumption = BashOperator(
        task_id="process_consumption",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/processing/process_consumption.py
        """,
    )

    fetch_smf = BashOperator(
        task_id="fetch_smf",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/ingestion/fetch_smf.py
        """,
    )

    process_smf = BashOperator(
        task_id="process_smf",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/processing/process_smf.py
        """,
    )

    process_market = BashOperator(
        task_id="process_market",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/processing/process_market.py
        """,
    )

    # =========================================================
    # AUXILIARY FEATURE BUILD (LATEST)
    # =========================================================
    build_generation_features_inference_latest = BashOperator(
        task_id="build_generation_features_inference_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline generation --mode inference_latest
        """,
    )

    build_consumption_features_inference_latest = BashOperator(
        task_id="build_consumption_features_inference_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline consumption --mode inference_latest
        """,
    )

    build_smf_features_inference_latest = BashOperator(
        task_id="build_smf_features_inference_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline smf --mode inference_latest
        """,
    )

    # =========================================================
    # AUXILIARY PREDICT (LATEST)
    # Uses pre-trained artifacts only
    # =========================================================
    predict_generation_latest = BashOperator(
        task_id="predict_generation_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/predict/predict_auxiliary_series.py \
          --pipeline generation \
          --mode latest \
          --run-id "{{{{ run_id }}}}"
        """,
    )

    predict_consumption_latest = BashOperator(
        task_id="predict_consumption_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/predict/predict_auxiliary_series.py \
          --pipeline consumption \
          --mode latest \
          --run-id "{{{{ run_id }}}}"
        """,
    )

    predict_smf_latest = BashOperator(
        task_id="predict_smf_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/predict/predict_auxiliary_series.py \
          --pipeline smf \
          --mode latest \
          --run-id "{{{{ run_id }}}}"
        """,
    )

    # =========================================================
    # MAIN PTF FEATURE BUILD (LATEST)
    # =========================================================
    build_features_inference_latest = BashOperator(
        task_id="build_features_inference_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline ptf --mode inference_latest
        """,
    )

    # =========================================================
    # MAIN PTF PREDICT
    # =========================================================
    predict_ptf_latest = BashOperator(
        task_id="predict_ptf_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/predict/predict_lgbm.py --mode latest --run-id "{{{{ run_id }}}}"
        """,
    )

    # =========================================================
    # DECISION / SIMULATION / EVAL
    # =========================================================
    generate_ptf_decision_signals_latest = BashOperator(
        task_id="generate_ptf_decision_signals_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/decision/generate_signals.py \
          --predictions-path data/predictions/ptf/ptf_predictions_history.parquet \
          --output-path data/decision/ptf/ptf_decision_signals.parquet \
          --summary-path data/decision/ptf/ptf_decision_summary.json \
          --mode latest_day
        """,
    )

    simulate_ptf_strategy_latest = BashOperator(
        task_id="simulate_ptf_strategy_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/decision/simulate_imbalance_cost.py \
          --decision-path data/decision/ptf/ptf_decision_signals.parquet \
          --market-path data/processed/market/market_data.parquet \
          --generation-path data/processed/generation/generation_processed.parquet \
          --output-path data/decision/ptf/ptf_strategy_simulation.parquet \
          --summary-path data/decision/ptf/ptf_strategy_simulation_summary.json \
          --mode latest_day \
          --base-generation 100 \
          --high-multiplier 1.20 \
          --low-multiplier 0.80 \
          --risky-high-multiplier 1.05 \
          --risky-low-multiplier 0.90 \
          --normal-multiplier 1.00 \
          --use-pred-as-ptf-fallback \
          --smf-from-ptf-multiplier 1.00
        """,
    )

    evaluate_ptf_latest = BashOperator(
        task_id="evaluate_ptf_latest",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/evalution/evaluate_ptf_forecast.py
        """,
    )

    # =========================================================
    # DEPENDENCIES
    # =========================================================
    fetch_epias >> process_ptf
    fetch_weather >> process_weather
    fetch_generation >> process_generation
    fetch_consumption >> process_consumption
    fetch_smf >> process_smf

    [process_ptf, process_smf] >> process_market

    process_generation >> build_generation_features_inference_latest
    process_consumption >> build_consumption_features_inference_latest
    process_smf >> build_smf_features_inference_latest

    build_generation_features_inference_latest >> predict_generation_latest
    build_consumption_features_inference_latest >> predict_consumption_latest
    build_smf_features_inference_latest >> predict_smf_latest

    [
        process_market,
        process_weather,
        predict_generation_latest,
        predict_consumption_latest,
        predict_smf_latest,
    ] >> build_features_inference_latest

    build_features_inference_latest >> predict_ptf_latest
    predict_ptf_latest >> generate_ptf_decision_signals_latest
    generate_ptf_decision_signals_latest >> simulate_ptf_strategy_latest
    simulate_ptf_strategy_latest >> evaluate_ptf_latest