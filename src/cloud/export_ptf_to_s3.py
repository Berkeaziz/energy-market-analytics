"""Explicit, repeatable export of local EPİAŞ PTF Parquet datasets to S3.

This module never connects to AWS unless the user runs the CLI without --dry-run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _validate_prefix(prefix: str) -> str:
    """Return a safe, normalized S3 prefix without directory traversal."""
    if (
        not prefix
        or prefix.startswith("/")
        or "\\" in prefix
        or any(segment in ("", ".", "..") for segment in prefix.split("/"))
    ):
        raise ValueError(
            "S3 prefix must be a relative, non-empty path without empty or '..' segments"
        )
    return prefix


def build_upload_plan(project_root: Path, prefix: str) -> list[tuple[Path, str]]:
    """Map existing PTF Parquet files to stable S3 object keys.

    Repeated exports target the same keys and therefore never accumulate
    additional per-run copies. No file is modified or deleted locally.
    """
    clean_prefix = _validate_prefix(prefix)
    root = Path(project_root)
    raw_root = root / "data/raw/epias/ptf_mcp"
    processed = root / "data/processed/ptf/ptf_processed.parquet"

    if not processed.is_file():
        raise FileNotFoundError(f"Required processed PTF dataset missing: {processed}")

    raw_files = sorted(path for path in raw_root.rglob("*.parquet") if path.is_file())
    if not raw_files:
        raise FileNotFoundError(f"No raw PTF Parquet files found in: {raw_root}")

    uploads = [
        (path, f"{clean_prefix}/raw/ptf/{path.relative_to(raw_root).as_posix()}")
        for path in raw_files
    ]
    uploads.append((processed, f"{clean_prefix}/curated/ptf/ptf_processed.parquet"))
    return uploads


def upload_files(plan: list[tuple[Path, str]], bucket: str, client: Any) -> int:
    """Upload all planned objects, propagating any failure to the caller."""
    if not bucket.strip():
        raise ValueError("S3 bucket must not be empty")

    count = 0
    for local_path, object_key in plan:
        client.upload_file(str(local_path), bucket, object_key)
        count += 1
        print(f"Uploaded s3://{bucket}/{object_key}")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export EPİAŞ PTF Parquet data to Amazon S3")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root with data/raw and data/processed directories",
    )
    parser.add_argument("--bucket", required=True, help="Existing private S3 bucket name")
    parser.add_argument("--prefix", default="energy-market", help="Relative S3 key prefix")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show planned uploads; do not initialize AWS or upload any data",
    )
    args = parser.parse_args(argv)

    plan = build_upload_plan(args.project_root, args.prefix)

    if args.dry_run:
        print(f"DRY RUN: {len(plan)} files would be uploaded to s3://{args.bucket}/")
        for local_path, object_key in plan:
            print(f"{local_path} -> {object_key}")
        return 0

    # Import lazily so dry-run works without boto3, credentials, or network.
    import boto3

    client = boto3.client("s3")
    count = upload_files(plan, args.bucket, client)
    print(f"Uploaded {count} PTF Parquet files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
