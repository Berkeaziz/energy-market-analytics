# AWS PTF S3 Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, testable S3 export for existing PTF Parquet datasets as the first milestone of the AWS Data Lake.

**Architecture:** Keep local Airflow and pandas processing unchanged. A new standalone Python CLI enumerates local Parquet files and maps them to deterministic private S3 object keys. Users can dry-run before explicitly uploading with their existing AWS identity. Manual Glue Catalog/Athena setup is documented; managed Glue ETL follows as a separately verifiable milestone.

**Tech Stack:** Python 3.10+, pathlib, argparse, boto3 (lazy-loaded), unittest, AWS S3/Glue Catalog/Athena.

**Spec:** `docs/superpowers/specs/2026-11-02-aws-ptf-data-lake-design.md`

## Global Constraints
- No AWS resources or charges created automatically; no credentials committed.
- Keep all current Airflow DAGs, data ingestion, model behavior and Docker Compose unchanged for Milestone 1.
- Work only in `feat/aws-ptf-data-lake`, not `main`.
- Map existing files, do not move or delete them.
- Use existing boto3 credential-chain resolution, not hardcoded access keys.
- Overwrite identical S3 keys on rerun rather than adding per-run duplicates.

## Review Focus
1. Empty source or absent processed PTF file: fail clearly before any upload.
2. Unexpected prefix (`..`, leading slash or empty after normalization): reject unsafe key layout.
3. Dry-run: no boto3 import, AWS identity resolution or network call.
4. Duplicate/repeated runs: deterministic S3 keys, no run-ID prefixes.
5. Upload failure: stop and expose error, never claim a partial upload succeeded.

---

### Task 1: S3 export planning and execution (test-first)

**Files:**
- Create: `tests/test_aws_ptf_s3_export.py`
- Create: `src/cloud/export_ptf_to_s3.py`
- Modify: `requirements.txt` (append `boto3`)

**Interfaces:**
- `build_upload_plan(project_root: Path, prefix: str) -> list[tuple[Path, str]]` takes a repo root, validates that the processed PTF Parquet file exists, and returns sorted raw/processed upload pairs.
- `upload_files(plan: list[tuple[Path, str]], bucket: str, client: object) -> int` uploads each pair with `client.upload_file(str(local_file), bucket, s3_key)`; return successful object count. Errors propagate.
- `main(argv: list[str] | None = None) -> int` parses `--project-root`, `--bucket`, `--prefix` (default `energy-market`) and `--dry-run`. Initialize boto3 only on the non-dry-run path.

- [ ] **Step 1: Write unit tests** for:
  - deterministic upload keys, e.g. `data/raw/epias/ptf_mcp/year=2026/month=11/chunk.parquet` maps to `energy-market/raw/ptf/year=2026/month=11/chunk.parquet`;
  - processed PTF maps to `energy-market/curated/ptf/ptf_processed.parquet`;
  - absent processed file raises `FileNotFoundError` before external calls;
  - bad prefix raises `ValueError`;
  - dry-run reports keys without constructing the AWS client;
  - real path with a fake client uploads exactly these S3 keys, returns count, and propagates failures.

- [ ] **Step 2: Run `python -m unittest discover -s tests -p 'test_aws_*.py' -v`**, confirm failure because implementation does not exist (not because the test is malformed).
- [ ] **Step 3: Implement the minimal CLI functions** in `src/cloud/export_ptf_to_s3.py`. Do not add automatic uploads to existing DAGs.
- [ ] **Step 4: Run the same unittest command**; require exit 0 and all new tests passing.
- [ ] **Step 5: Run a CLI dry-run** against a controlled fixture directory. Confirm no AWS credentials required.
- [ ] **Step 6: Review output and commit** tests, module and dependency.

### Task 2: Private AWS setup and usage documentation

**Files:**
- Create: `docs/aws-ptf-data-lake.md`
- Modify: `README.md` (short link and status marker)

**Interfaces:** human-readable instructions, no automatic provisioning.

- [ ] **Step 1: Document prerequisites** (AWS account with explicit spending controls, AWS CLI credentials on host, Python dependencies).
- [ ] **Step 2: Document private S3 bucket configuration**, encryption, public-access blocking, least-privilege S3 upload/list permissions, region selection and object prefixes.
- [ ] **Step 3: Document dry-run and explicit upload commands** using placeholder names; never place actual account IDs or credentials in git.
- [ ] **Step 4: Document Glue crawler configuration** for curated PTF Parquet and Athena workgroup/results bucket; include one example hourly aggregation query after verifying schema names in the user's account.
- [ ] **Step 5: Add cost cleanup instructions**, how to remove any manually created Glue/Athena resources, and a note that AWS is *not deployed* until cloud validation is completed.
- [ ] **Step 6: Re-run all tests** and commit documentation.

### Follow-on milestone (separate plan after S3/Athena validation)

An AWS Glue ETL job transforms raw Parquet to year/month-partitioned curated datasets and is optionally triggered from Airflow. It must have independent tests comparing record counts, hourly uniqueness and sample PTF values against `src/processing/process_ptf.py`. Do not implement it before validating Task 1 and the cloud data path.
