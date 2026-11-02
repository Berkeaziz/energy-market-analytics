# AWS PTF Data Lake — Milestone 1

**Status:** Code-ready for an optional S3 export. **Not yet deployed or tested against a real AWS account.** The existing local Airflow, EPİAŞ ingestion, MLflow, forecasting and Streamlit pipeline remain unchanged.

## Data flow

EPİAŞ API (existing local Airflow/Python) → local raw PTF Parquet → private Amazon S3 (`energy-market/raw/ptf/`)

Existing local processing → `data/processed/ptf/ptf_processed.parquet` → private Amazon S3 (`energy-market/curated/ptf/`) → AWS Glue Data Catalog → Athena SQL

The first curated PTF export is one Parquet file. A managed AWS Glue transformation to partitioned curated data is a **later milestone** and must not be claimed as implemented.

## Prerequisites

1. Use **your own** AWS account. Previously expired AWS credits should not be assumed to be renewed. Check your current billing account and [AWS Free Tier eligibility](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier.html).
2. Before provisioning, set up [AWS Budgets alerts](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-create.html). Budget email alerts are **not a hard spending cap**. S3, Glue and Athena can incur charges.
3. Choose one AWS region for S3, Glue and Athena; keep all resources in that same region.
4. Create an S3 bucket with **Block Public Access enabled** and server-side encryption enabled. Do not turn on public website hosting or public object ACLs.
5. Configure an AWS CLI identity/profile outside this repository, with only the necessary access. Do not store AWS keys, EPİAŞ passwords, or account IDs in Git or screenshots.

Use a dedicated IAM identity instead of the root account. Example minimal upload policy (replace the placeholder bucket, and scope to your prefix):

    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": ["s3:ListBucket"],
          "Resource": "arn:aws:s3:::YOUR_PRIVATE_BUCKET"
        },
        {
          "Effect": "Allow",
          "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
          "Resource": "arn:aws:s3:::YOUR_PRIVATE_BUCKET/energy-market/*"
        }
      ]
    }

Additional AWS CLI verification or Glue/Athena access may require separate read/catalog permissions. Grant only what the actual task needs.

## Local setup

The data directory is deliberately not committed to GitHub. First run the project's existing local ingestion and processing so these files are present:

- `data/raw/epias/ptf_mcp/**/*.parquet`
- `data/processed/ptf/ptf_processed.parquet`

Install the project's dependencies (or just `boto3` for the exporter):

    python -m pip install boto3

### Dry-run (no AWS access or charges)

    python -m src.cloud.export_ptf_to_s3 --bucket YOUR_PRIVATE_BUCKET --dry-run

The CLI prints which local files would be uploaded and the target S3 object keys. It does not initialize boto3 or contact AWS in dry-run mode.

### Explicit upload (creates S3 objects and may incur costs)

After confirming bucket policy, identity, cost controls, the data-sharing terms and the dry-run:

    python -m src.cloud.export_ptf_to_s3 --bucket YOUR_PRIVATE_BUCKET

You can optionally pass `--project-root PATH` and `--prefix energy-market`.

Key layout (examples):

    s3://YOUR_PRIVATE_BUCKET/energy-market/raw/ptf/year=2026/month=11/chunk.parquet
    s3://YOUR_PRIVATE_BUCKET/energy-market/curated/ptf/ptf_processed.parquet

The exporter uses deterministic object keys, so rerunning the same dataset overwrites the same keys instead of generating extra run-ID copies. It does not delete S3 data; if old raw files disappear locally, their old S3 copies remain until manually cleaned up. Partial uploads raise an error and require inspection/retry.

## Glue Data Catalog and Athena (manual, after S3 upload)

1. In the AWS Glue console, create a database (for example, `energy_market`).
2. Configure a crawler to scan only `s3://YOUR_PRIVATE_BUCKET/energy-market/curated/ptf/`. Use the same region as the bucket. Running a Glue crawler may incur costs.
3. Run the crawler, then inspect the discovered table name and schema. Check the `date` field's inferred timestamp type and numeric PTF columns. Do not assume a table name until the crawler completes.
4. Configure an Athena workgroup with a **private query-results S3 location** and, where supported, a per-query scan limit. Athena queries and result storage may incur costs.
5. In Athena, select `energy_market` and the crawler-created table. Run small validation queries; for example, after replacing the table placeholder:

       SELECT COUNT(*) AS n_rows FROM energy_market.YOUR_CRAWLED_TABLE;
       SELECT MIN(date) AS first_hour, MAX(date) AS last_hour
       FROM energy_market.YOUR_CRAWLED_TABLE;
       SELECT date, ptf FROM energy_market.YOUR_CRAWLED_TABLE
       ORDER BY date DESC LIMIT 10;

6. Compare results against `data/processed/ptf/ptf_processed.parquet`: count, min/max timestamp, unique hourly records and selected PTF values. A successful local dry-run alone is not cloud validation.

## Validation without cloud credentials

    python -m unittest discover -s tests -p 'test_aws_*.py' -v

Tests cover deterministic key names, missing inputs, prefix validation, dry-run isolation and upload error propagation.

## Cost controls and clean-up

- Keep the S3 bucket private. Set billing alerts before running a Glue crawler or Athena SQL.
- Prefer small queries; query only what is necessary.
- Stop scheduled Glue crawlers and remove manually created catalog resources when finished.
- Athena writes result objects; inspect and clean up unwanted query-results files, following your retention requirements.
- Delete S3 data and buckets only after confirming there is nothing you need to keep. This CLI never deletes anything.

## Future milestone

Implement an AWS Glue ETL job to transform raw PTF records into year/month-partitioned Parquet in S3; compare its results with local `src/processing/process_ptf.py`; then optionally add an Airflow trigger. Track this separately from the already testable uploader.
