# MSTS 압축 텔레메트리 매니페스트 생성을 검증하는 테스트
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.data.msts_manifest import build_msts_manifest


class MstsManifestTest(unittest.TestCase):
    def test_manifest_reads_telemetry_header_and_sampling_interval_from_zip(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            archive_path = root / "210303_FULL.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "run/GT01_20210303_145747.txt",
                    "#time ACCEL RATE BIT\n0.00 1.0 0\n0.01 1.1 1\n",
                )
                archive.writestr("run/readme.bin", b"metadata")

            manifest = build_msts_manifest(root)

        self.assertEqual(manifest["summary"]["archive_count"], 1)
        self.assertEqual(manifest["summary"]["entry_count"], 2)
        telemetry = manifest["entries"][0]
        self.assertEqual(telemetry["telemetry_group"], "GT01")
        self.assertEqual(telemetry["run_id"], "20210303_145747")
        self.assertEqual(telemetry["column_count"], 4)
        self.assertEqual(telemetry["columns"], ["time", "ACCEL", "RATE", "BIT"])
        self.assertAlmostEqual(telemetry["sample_interval_seconds"], 0.01)
        self.assertEqual(telemetry["time_status"], "increasing")

    def test_manifest_preserves_non_telemetry_entries_without_parsing_them(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            archive_path = root / "210513_FULL.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("run/LASM_20210513_raw.dat", b"raw")

            manifest = build_msts_manifest(root)

        entry = manifest["entries"][0]
        self.assertEqual(entry["file_type"], "dat")
        self.assertIsNone(entry["column_count"])
        self.assertEqual(manifest["summary"]["telemetry_entry_count"], 0)

    def test_manifest_inspects_extracted_telemetry_text_files(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            telemetry_path = root / "_dataanalyzer_validation" / "GT01_run.txt"
            telemetry_path.parent.mkdir()
            telemetry_path.write_text(
                "#time ACCEL BIT\n0.00 1.0 0\n0.01 1.1 1\n",
                encoding="utf-8",
            )

            manifest = build_msts_manifest(root)

        entry = manifest["entries"][0]
        self.assertEqual(entry["container_type"], "filesystem")
        self.assertEqual(entry["telemetry_group"], "GT01")
        self.assertEqual(entry["column_count"], 3)
        self.assertAlmostEqual(entry["sample_interval_seconds"], 0.01)


if __name__ == "__main__":
    unittest.main()
