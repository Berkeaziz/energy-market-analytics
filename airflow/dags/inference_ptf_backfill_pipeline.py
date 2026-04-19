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
    dag_id="inference_ptf_backfill_pipeline",
    description=(
        "PTF backfill inference pipeline with auxiliary forecasts: "
        "fetch/process raw data -> "
        "aux backfill feature build -> aux backfill predict -> "
        "fill missing externals in ptf backfill features -> "
        "predict ptf backfill -> decision -> simulate -> evaluate"
    ),
    start_date=pendulum.datetime(2026, 4, 1, tz=LOCAL_TZ),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    default_args=default_args,
    tags=[
        "ptf",
        "inference",
        "backfill",
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
    # AUXILIARY FEATURE BUILD (BACKFILL)
    # =========================================================
    build_generation_features_inference_backfill = BashOperator(
        task_id="build_generation_features_inference_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline generation --mode inference_backfill
        """,
    )

    build_consumption_features_inference_backfill = BashOperator(
        task_id="build_consumption_features_inference_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline consumption --mode inference_backfill
        """,
    )

    build_smf_features_inference_backfill = BashOperator(
        task_id="build_smf_features_inference_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline smf --mode inference_backfill
        """,
    )

    # =========================================================
    # AUXILIARY PREDICT (BACKFILL)
    # Uses pre-trained artifacts only
    # =========================================================
    predict_generation_backfill = BashOperator(
        task_id="predict_generation_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail

        MODE="{{{{ dag_run.conf.get('mode', 'backfill_auto') if dag_run else 'backfill_auto' }}}}"
        START_DATE="{{{{ dag_run.conf.get('start_date', '') if dag_run else '' }}}}"
        END_DATE="{{{{ dag_run.conf.get('end_date', '') if dag_run else '' }}}}"

        case "$MODE" in
          backfill_auto|backfill_full|backfill_range)
            ;;
          *)
            echo "ERROR: invalid mode: $MODE"
            exit 1
            ;;
        esac

        CMD='{PYTHON_BIN} src/predict/predict_auxiliary_series.py --pipeline generation --mode '"$MODE"' --run-id "{{{{ run_id }}}}"'

        if [ "$MODE" = "backfill_range" ]; then
          if [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
            echo "ERROR: start_date and end_date are required for backfill_range"
            exit 1
          fi
          CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
        fi

        echo "Running command: $CMD"
        eval "$CMD"
        """,
    )

    predict_consumption_backfill = BashOperator(
        task_id="predict_consumption_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail

        MODE="{{{{ dag_run.conf.get('mode', 'backfill_auto') if dag_run else 'backfill_auto' }}}}"
        START_DATE="{{{{ dag_run.conf.get('start_date', '') if dag_run else '' }}}}"
        END_DATE="{{{{ dag_run.conf.get('end_date', '') if dag_run else '' }}}}"

        case "$MODE" in
          backfill_auto|backfill_full|backfill_range)
            ;;
          *)
            echo "ERROR: invalid mode: $MODE"
            exit 1
            ;;
        esac

        CMD='{PYTHON_BIN} src/predict/predict_auxiliary_series.py --pipeline consumption --mode '"$MODE"' --run-id "{{{{ run_id }}}}"'

        if [ "$MODE" = "backfill_range" ]; then
          if [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
            echo "ERROR: start_date and end_date are required for backfill_range"
            exit 1
          fi
          CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
        fi

        echo "Running command: $CMD"
        eval "$CMD"
        """,
    )

    predict_smf_backfill = BashOperator(
        task_id="predict_smf_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail

        MODE="{{{{ dag_run.conf.get('mode', 'backfill_auto') if dag_run else 'backfill_auto' }}}}"
        START_DATE="{{{{ dag_run.conf.get('start_date', '') if dag_run else '' }}}}"
        END_DATE="{{{{ dag_run.conf.get('end_date', '') if dag_run else '' }}}}"

        case "$MODE" in
          backfill_auto|backfill_full|backfill_range)
            ;;
          *)
            echo "ERROR: invalid mode: $MODE"
            exit 1
            ;;
        esac

        CMD='{PYTHON_BIN} src/predict/predict_auxiliary_series.py --pipeline smf --mode '"$MODE"' --run-id "{{{{ run_id }}}}"'

        if [ "$MODE" = "backfill_range" ]; then
          if [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
            echo "ERROR: start_date and end_date are required for backfill_range"
            exit 1
          fi
          CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
        fi

        echo "Running command: $CMD"
        eval "$CMD"
        """,
    )

    # =========================================================
    # MAIN PTF FEATURE BUILD (BACKFILL)
    # =========================================================
    build_features_inference_backfill = BashOperator(
        task_id="build_features_inference_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline ptf --mode inference_backfill
        """,
    )

    # =========================================================
    # MAIN PTF PREDICT
    # =========================================================
    predict_ptf_backfill = BashOperator(
        task_id="predict_ptf_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail

        MODE="{{{{ dag_run.conf.get('mode', 'backfill_auto') if dag_run else 'backfill_auto' }}}}"
        START_DATE="{{{{ dag_run.conf.get('start_date', '') if dag_run else '' }}}}"
        END_DATE="{{{{ dag_run.conf.get('end_date', '') if dag_run else '' }}}}"

        case "$MODE" in
          backfill_auto|backfill_full|backfill_range)
            ;;
          *)
            echo "ERROR: invalid mode: $MODE"
            echo "Allowed modes: backfill_auto, backfill_full, backfill_range"
            exit 1
            ;;
        esac

        CMD='{PYTHON_BIN} src/predict/predict_lgbm.py --mode '"$MODE"' --run-id "{{{{ run_id }}}}"'

        if [ "$MODE" = "backfill_range" ]; then
          if [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
            echo "ERROR: start_date and end_date are required for backfill_range"
            exit 1
          fi
          CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
        fi

        echo "Running command: $CMD"
        eval "$CMD"
        """,
    )

    # =========================================================
    # DECISION / SIMULATION / EVAL
    # =========================================================
    generate_ptf_decision_signals_backfill = BashOperator(
        task_id="generate_ptf_decision_signals_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail

        DECISION_MODE="all"
        START_TIME="{{{{ dag_run.conf.get('start_time', '') if dag_run else '' }}}}"
        END_TIME="{{{{ dag_run.conf.get('end_time', '') if dag_run else '' }}}}"

        CMD="{PYTHON_BIN} src/decision/generate_signals.py \
          --predictions-path data/predictions/ptf/ptf_predictions_history.parquet \
          --output-path data/decision/ptf/ptf_decision_signals.parquet \
          --summary-path data/decision/ptf/ptf_decision_summary.json \
          --mode $DECISION_MODE"

        if [ -n "$START_TIME" ] && [ -n "$END_TIME" ]; then
          CMD="$CMD --mode range --start-time '$START_TIME' --end-time '$END_TIME'"
        fi

        echo "Running command: $CMD"
        eval "$CMD"
        """,
    )

    simulate_ptf_strategy_backfill = BashOperator(
        task_id="simulate_ptf_strategy_backfill",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail

        START_TIME="{{{{ dag_run.conf.get('start_time', '') if dag_run else '' }}}}"
        END_TIME="{{{{ dag_run.conf.get('end_time', '') if dag_run else '' }}}}"

        CMD="{PYTHON_BIN} src/decision/simulate_imbalance_cost.py \
          --decision-path data/decision/ptf/ptf_decision_signals.parquet \
          --market-path data/processed/market/market_data.parquet \
          --generation-path data/processed/generation/generation_processed.parquet \
          --output-path data/decision/ptf/ptf_strategy_simulation.parquet \
          --summary-path data/decision/ptf/ptf_strategy_simulation_summary.json \
          --mode all \
          --base-generation 100 \
          --high-multiplier 1.20 \
          --low-multiplier 0.80 \
          --risky-high-multiplier 1.05 \
          --risky-low-multiplier 0.90 \
          --normal-multiplier 1.00 \
          --use-pred-as-ptf-fallback \
          --smf-from-ptf-multiplier 1.00"

        if [ -n "$START_TIME" ] && [ -n "$END_TIME" ]; then
          CMD="$CMD --mode range --start-time '$START_TIME' --end-time '$END_TIME'"
        fi

        echo "Running command: $CMD"
        eval "$CMD"
        """,
    )

    evaluate_ptf_backfill = BashOperator(
        task_id="evaluate_ptf_backfill",
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

    process_generation >> build_generation_features_inference_backfill
    process_consumption >> build_consumption_features_inference_backfill
    process_smf >> build_smf_features_inference_backfill

    build_generation_features_inference_backfill >> predict_generation_backfill
    build_consumption_features_inference_backfill >> predict_consumption_backfill
    build_smf_features_inference_backfill >> predict_smf_backfill

    [
        process_market,
        process_weather,
        predict_generation_backfill,
        predict_consumption_backfill,
        predict_smf_backfill,
    ] >> build_features_inference_backfill

    build_features_inference_backfill >> predict_ptf_backfill
    predict_ptf_backfill >> generate_ptf_decision_signals_backfill
    generate_ptf_decision_signals_backfill >> simulate_ptf_strategy_backfill
    simulate_ptf_strategy_backfill >> evaluate_ptf_backfill