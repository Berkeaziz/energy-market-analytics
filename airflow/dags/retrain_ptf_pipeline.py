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
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="retrain_ptf_pipeline",
    description=(
        "PTF retraining pipeline with auxiliary models: "
        "fetch/process -> aux feature build -> aux train -> "
        "ptf feature build -> ptf train"
    ),
    start_date=pendulum.datetime(2026, 4, 1, tz=LOCAL_TZ),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    default_args=default_args,
    tags=["ptf", "retrain", "ml", "weather", "generation", "consumption", "smf"],
    render_template_as_native_obj=True,
) as dag:

    # =========================================================
    # FETCH / PROCESS
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

    # =========================================================
    # AUX FEATURE BUILD (TRAIN)
    # =========================================================
    build_generation_features = BashOperator(
        task_id="build_generation_features",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline generation --mode train
        """,
    )

    build_consumption_features = BashOperator(
        task_id="build_consumption_features",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline consumption --mode train
        """,
    )

    build_smf_features = BashOperator(
        task_id="build_smf_features",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline smf --mode train
        """,
    )

    # =========================================================
    # AUX TRAIN
    # =========================================================
    train_generation = BashOperator(
        task_id="train_generation",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/forecasting/ptf/train_gen.py
        """,
    )

    train_consumption = BashOperator(
        task_id="train_consumption",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/forecasting/ptf/train_cons.py
        """,
    )

    train_smf = BashOperator(
        task_id="train_smf",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/forecasting/ptf/train_smf.py
        """,
    )

    # =========================================================
    # MAIN PTF FEATURE BUILD + TRAIN
    # =========================================================
    build_features_train = BashOperator(
        task_id="build_features_train",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/features/build_ptf_features.py --pipeline ptf --mode train
        """,
    )

    train_lgbm = BashOperator(
        task_id="train_lgbm",
        cwd=PROJECT_DIR,
        bash_command=f"""
        set -euo pipefail
        {PYTHON_BIN} src/forecasting/ptf/train_lgbm.py
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

    # AUX FEATURE BUILD
    process_generation >> build_generation_features
    process_consumption >> build_consumption_features
    process_smf >> build_smf_features

    # AUX TRAIN
    build_generation_features >> train_generation
    build_consumption_features >> train_consumption
    build_smf_features >> train_smf

    # MAIN FEATURE BUILD waits for everything
    [
        process_ptf,
        process_weather,
        train_generation,
        train_consumption,
        train_smf,
    ] >> build_features_train

    build_features_train >> train_lgbm