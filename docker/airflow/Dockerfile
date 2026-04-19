FROM apache/airflow:2.9.3-python3.11

USER root

ENV TZ=Europe/Istanbul

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    pyarrow \
    requests \
    python-dotenv \
    scikit-learn \
    lightgbm \
    mlflow \
    psycopg2-binary \
    joblib \
    pendulum