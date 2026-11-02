# AWS PTF Data Lake — Design (2026-11-02)

## Goal
Extend the existing Energy Market Analytics portfolio project with a reproducible, optional AWS data-lake path, without breaking local Airflow, MLflow, forecasting, or Streamlit. This demonstrates real AWS S3, Glue Data Catalog and Athena skills, then adds managed Glue ETL in a second milestone.

## Existing system
- EPİAŞ ingestion: `src/ingestion/fetch_epias_ptf.py`, producing partitioned Parquet under `data/raw/epias/ptf_mcp/`.
- PTF processing: `src/processing/process_ptf.py`, producing `data/processed/ptf/ptf_processed.parquet`.
- `airflow/dags/inference_ptf_latest_pipeline.py` runs hourly; retraining/backfill are separate DAGs.
- EPİAŞ credentials currently remain in local Airflow environment. No existing AWS integration.

## Approaches considered
1. **Phased, low-risk integration — chosen:** an explicit CLI uploads existing raw and processed PTF Parquet data to private S3; Glue Data Catalog discovers the datasets; Athena queries them. The local pipeline remains unchanged. Add an AWS Glue ETL job and Airflow orchestration once this path is verified.
2. Full immediate migration: move API ingestion, processing and scheduling to Lambda/Glue/managed orchestration. Offers deeper AWS ownership but introduces credentials migration, runtime charges and operational risk before the initial data path is proven.

## Milestone 1: S3 → Glue Catalog → Athena
- A standalone Python CLI plans and executes deterministic uploads, mapping `data/raw/epias/ptf_mcp/**/*.parquet` to `<prefix>/raw/ptf/...` and `data/processed/ptf/ptf_processed.parquet` to `<prefix>/curated/ptf/ptf_processed.parquet`.
- Existing local files remain unchanged. Re-uploading a file with the same name overwrites its object key. EPİAŞ ingestion may create newly named raw Parquet files with overlapping hourly observations; the future Glue ETL must deduplicate by timestamp. No automated deletion.
- `--dry-run` validates paths and prints the upload plan without AWS authentication or network operations.
- No AWS upload is made automatically on import, Docker startup or DAG parse. A user explicitly runs the CLI.
- Use boto3's normal credentials provider chain; never store API keys or EPİAŞ passwords in source, Docker images or documentation.
- Document creating a private, encrypted S3 bucket, least-privilege IAM access, Glue crawler and Athena workgroup/output location. Require spend controls before running paid AWS resources.
- Dataset caveat: the first curated PTF export is a single Parquet file. Partitioned managed transformations are Milestone 2, not falsely claimed in Milestone 1.

## Milestone 2: AWS Glue ETL
- Read raw PTF Parquet from S3, normalize required columns and dates, deduplicate hourly records, partition curated output by year/month in Parquet and expose it through Glue Catalog/Athena.
- Keep the current local pandas implementation as the reference. Validate Glue output counts, duplicate timestamps, and sample prices against the local result.
- Hook AWS export/Glue tasks into Airflow only after manual run succeeds. Make the cloud path optional and disabled by default.

## Constraints
- Do not create billable AWS resources or use third-party family accounts for promotional credits in code.
- Preserve all existing three Airflow DAGs and model behavior for Milestone 1.
- Do not overwrite `main`; work on a dedicated feature branch.
- Never commit secrets, real AWS account identifiers, or user data dumps.
- Tests must cover empty/missing files, dry-run, deterministic keys, upload failure propagation and no accidental credential/network use.

## Acceptance for Milestone 1
- A reproducible dry-run lists exactly the expected raw files and curated output.
- Unit tests pass without AWS credentials and without contacting AWS.
- With the user's own configured AWS account and explicit approval for resource creation: S3 objects upload, Glue sees the schema and Athena retrieves an expected PTF sample.
- Only *after* live verification should the CV or README describe AWS deployment as completed.
