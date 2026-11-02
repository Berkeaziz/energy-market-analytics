# Low-Cost AWS Data Lake — Energy Market Analytics

**Date:** 2026-11-02  
**Repository:** `Berkeaziz/energy-market-analytics`  
**Branch:** `feat/aws-low-cost-lake`  
**Status:** Design proposal, pending owner review

## Intent

Add a real, demonstrable AWS data-engineering layer to the existing Turkish electricity-market forecasting project for Berke's Data Engineer portfolio. Keep the existing EPİAŞ ingestion, hourly Airflow inference, LightGBM forecasting, MLflow experiments, and Streamlit dashboard functioning locally. The cloud layer should be easy to demonstrate and cheap to leave idle after AWS promotional credits expire.

## Goals

1. Safely store existing EPİAŞ PTF data as partitioned raw Parquet in a **private S3 data lake**.
2. Run a **manually invoked AWS Glue Spark ETL** job that cleans and deduplicates PTF, then writes partitioned curated Parquet back to S3.
3. Query the curated data using **Athena SQL** with a **Glue Data Catalog** table.
4. Demonstrate repeatable processing, date/time semantics, data-quality checks, and readable documentation/screenshots for GitHub/CV.
5. Keep normal idle cost near zero and the modest interactive demo below a **target of $5 per month** in AWS service charges, without representing an AWS Budget alert as a hard cap.

## Non-goals

- No continuously running EC2, RDS, managed Airflow, Spark cluster, load balancer, NAT gateway, or public cloud-hosted dashboard.
- No cloud-based retraining, MLflow hosting, or replacement of the existing forecasting pipeline in v1.
- No AWS resource creation, deployed job execution, or upload of real EPİAŞ data until the owner separately confirms the AWS account/region and cost implications.
- No committed API credentials, AWS keys, raw bulk datasets, local `.env`, or sensitive configuration.

## Starting state

The existing repository has a Docker Compose stack with Airflow, MLflow, and PostgreSQL. Its `inference_ptf_latest_pipeline` runs hourly with local file paths such as `data/raw/epias/ptf_mcp` and `data/processed/ptf/ptf_processed.parquet`. Ingestion authenticates to EPİAŞ using environment variables. Existing PTF processing standardizes date/PTF fields and resolves duplicate timestamps. These behaviors should be preserved and the new cloud workflow should be opt-in, not a breaking refactor.

## Approaches considered

- **Selected: Local orchestration + on-demand serverless AWS data lake.** Reuse local ingestion, add an explicit S3 upload, run Glue only during demos, and query Athena. Lowest idle bill and clear AWS experience; less suitable for 24/7 production workloads.
- **Rejected for v1: Full cloud stack.** Move Airflow, MLflow, dashboard, and databases to persistent AWS compute. Strong live-demo story but recurring compute and storage cost.
- **Rejected: Local-only analytics with screenshots.** Cheapest, but does not demonstrate actual Glue, S3, or Athena integration.

## Proposed data flow

1. Existing local EPİAŞ PTF ingestion produces Parquet. A separate explicit CLI command uploads selected partitions to `s3://<bucket>/raw/epias/ptf_mcp/` with no public ACLs.
2. An on-demand Glue Spark job reads PTF raw Parquet, normalizes `date`/`ptf`, drops invalid rows, resolves duplicate timestamps consistently with existing `process_ptf.py`, and writes compressed curated Parquet partitioned by `year` and `month` to `s3://<bucket>/curated/ptf/`.
3. A Glue Data Catalog table is created/updated for the curated layout (initially one-time/manual, avoiding recurring crawler runs). Athena SQL verifies row count, date coverage, duplicate-free timestamps, and average PTF grouped by month.
4. The LightGBM, MLflow, Airflow and Streamlit paths remain unchanged. Cloud artifacts can be consumed later as an optional extension, not during v1.

## Boundaries and interfaces

- Place cloud-only scripts and deploy templates under `aws/`; keep provider-agnostic ingestion/processing in their current `src/` modules.
- A small Python uploader CLI accepts a local Parquet root, bucket, prefix and `--dry-run` option; it must never assume credentials or automatically create resources.
- The Glue job receives source and destination S3 URIs as parameters. Re-running it over the same data must not inflate duplicates or produce stale partitions.
- Store region and names in explicit configuration examples; default target is `eu-central-1` (Frankfurt), subject to checking current account resources and pricing. Do not hardcode account IDs or secrets.

## Safety and costs

- S3 bucket: block all public access, encryption on, narrow IAM actions. Use AWS CLI profile/role-based access; no static keys in source.
- Only one modest sample upload and one test Glue run initially. No automatic recurring AWS schedules in v1.
- Set a $5 monthly AWS Budget **notification** before live execution. Budget notifications are not strict spending limits. Check region and account for pre-existing billable resources.
- Athena queries target partitioned Parquet and avoid scanning unnecessary columns or partitions.
- Clean up temporary Glue jobs/crawlers and sample resources when no longer needed; preserve the documented pipeline in GitHub.

## Verification / acceptance

- Unit tests: S3 key mapping, dry-run no-write behavior, invalid input rejection, timestamp/duplicate cleaning parity, and idempotent output layout using small fixtures.
- Local lint/test runs pass without AWS credentials; the current application remains intact.
- AWS smoke test, only after account setup authorization: upload test Parquet, run Glue job once, execute Athena SQL, verify expected counts and no duplicate timestamps, inspect Billing.
- README gains an architecture diagram, setup instructions, the sample Athena SQL and observed costs, with no secret material.

## Delivery sequence

1. Cloud uploader and unit tests with local dry-run.
2. Glue Spark ETL script, small fixture tests, Athena DDL/SQL, documentation.
3. Review costs and configure AWS identity/bucket/budget with the owner.
4. One authorized live demonstration, collect evidence and update README.

## Open implementation decisions

The implementer may choose exact module names and testing helpers consistent with the repository. Ask the owner before any actual AWS billable resource creation or live data transfer; confirm whether a suitable existing S3 bucket is available. 
