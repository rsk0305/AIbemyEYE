# MSTS 전체 텔레메트리 스트리밍 품질 검사를 검증하는 테스트
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src.data.msts_manifest import build_msts_manifest
from src.data.msts_quality import scan_msts_telemetry


class MstsQualityTest(unittest.TestCase):
    def test_quality_scan_counts_rows_checks_time_and_samples_core_channels(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "210303_FULL.zip", "w") as archive:
                archive.writestr(
                    "run/GT01_1_20210303_145747.txt",
                    "#time POWER RATE OTHER\n"
                    "0.00 1.0 2.0 0\n"
                    "0.01 1.1 2.1 0\n"
                    "0.02 1.2 2.2 0\n",
                )

            quality, samples = scan_msts_telemetry(
                root,
                build_msts_manifest(root),
                core_channels=("POWER", "RATE"),
                sample_stride=2,
            )

        entry = quality["entries"][0]
        self.assertEqual(entry["row_count"], 3)
        self.assertEqual(entry["time_regressions"], 0)
        self.assertEqual(entry["width_mismatches"], 0)
        self.assertEqual(len(samples), 1)
        np.testing.assert_allclose(
            samples[0]["values"],
            np.array([[1.0, 2.0], [1.2, 2.2]]),
        )

    def test_quality_scan_excludes_files_without_every_core_channel(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "210513_FULL.zip", "w") as archive:
                archive.writestr(
                    "run/GT01_1_20210513_103047.txt",
                    "#time POWER\n0.00 1.0\n",
                )

            quality, samples = scan_msts_telemetry(
                root,
                build_msts_manifest(root),
                core_channels=("POWER", "RATE"),
            )

        self.assertEqual(quality["summary"]["core_complete_entry_count"], 0)
        self.assertEqual(samples, [])
