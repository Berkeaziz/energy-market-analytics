from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cloud.export_ptf_to_s3 import build_upload_plan, main, upload_files


class AwsPtfExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def make_files(self) -> tuple[Path, Path]:
        raw = self.root / "data/raw/epias/ptf_mcp/year=2026/month=11/chunk.parquet"
        processed = self.root / "data/processed/ptf/ptf_processed.parquet"
        raw.parent.mkdir(parents=True, exist_ok=True)
        processed.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"raw-data")
        processed.write_bytes(b"processed-data")
        return raw, processed

    def test_build_plan_preserves_raw_partitions_and_curated_key(self) -> None:
        raw, processed = self.make_files()
        plan = build_upload_plan(self.root, "energy-market")
        self.assertEqual(
            plan,
            [
                (raw, "energy-market/raw/ptf/year=2026/month=11/chunk.parquet"),
                (processed, "energy-market/curated/ptf/ptf_processed.parquet"),
            ],
        )

    def test_build_plan_rejects_missing_processed_file(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "ptf_processed.parquet"):
            build_upload_plan(self.root, "energy-market")

    def test_build_plan_rejects_missing_raw_files(self) -> None:
        processed = self.root / "data/processed/ptf/ptf_processed.parquet"
        processed.parent.mkdir(parents=True)
        processed.write_bytes(b"processed-data")
        with self.assertRaisesRegex(FileNotFoundError, "raw"):
            build_upload_plan(self.root, "energy-market")

    def test_build_plan_rejects_unsafe_prefix(self) -> None:
        self.make_files()
        for prefix in ("", "/absolute", "../traversal", "safe/../bad"):
            with self.subTest(prefix=prefix):
                with self.assertRaises(ValueError):
                    build_upload_plan(self.root, prefix)

    def test_dry_run_does_not_load_boto3(self) -> None:
        self.make_files()
        output = io.StringIO()
        with patch.dict(sys.modules, {"boto3": None}):
            with contextlib.redirect_stdout(output):
                result = main(
                    ["--project-root", str(self.root), "--bucket", "demo-bucket", "--dry-run"]
                )
        self.assertEqual(result, 0)
        self.assertIn("energy-market/curated/ptf/ptf_processed.parquet", output.getvalue())

    def test_upload_files_uses_deterministic_keys_and_returns_count(self) -> None:
        self.make_files()
        plan = build_upload_plan(self.root, "energy-market")

        class Client:
            def __init__(self) -> None:
                self.calls = []

            def upload_file(self, filename: str, bucket: str, key: str) -> None:
                self.calls.append((filename, bucket, key))

        client = Client()
        count = upload_files(plan, "demo-bucket", client)
        self.assertEqual(count, 2)
        self.assertEqual([call[2] for call in client.calls], [key for _, key in plan])
        self.assertEqual({call[1] for call in client.calls}, {"demo-bucket"})

    def test_upload_failure_is_not_suppressed(self) -> None:
        self.make_files()
        plan = build_upload_plan(self.root, "energy-market")

        class FailingClient:
            def upload_file(self, filename: str, bucket: str, key: str) -> None:
                raise RuntimeError("S3 upload failed")

        with self.assertRaisesRegex(RuntimeError, "S3 upload failed"):
            upload_files(plan, "demo-bucket", FailingClient())


if __name__ == "__main__":
    unittest.main()
