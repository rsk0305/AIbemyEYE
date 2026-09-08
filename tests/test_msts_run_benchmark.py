# MSTS 시험 실행 단위의 정상 학습 및 시나리오 검증을 검증하는 테스트
import unittest

import numpy as np

from src.utils.msts_run_benchmark import run_msts_run_benchmark


class MstsRunBenchmarkTest(unittest.TestCase):
    def test_benchmark_keeps_run_groups_disjoint_and_detects_sustained_fault(self):
        timeline = np.linspace(0.0, 12.0, 180)
        normal = np.column_stack([np.sin(timeline), np.cos(timeline)])
        fault = normal.copy()
        fault[100:110] += 20.0
        runs = {
            "train": normal,
            "calibration": normal * 1.01,
            "normal_test": normal * 0.99,
            "fault_test": fault,
            "unlabeled": normal * 1.02,
        }

        result = run_msts_run_benchmark(
            runs,
            train_run_ids=("train",),
            calibration_run_ids=("calibration",),
            labeled_test_labels={"normal_test": 0, "fault_test": 1},
            unlabeled_run_ids=("unlabeled",),
            window_size=33,
            quantile=0.99,
            min_consecutive_alarms=3,
        )

        self.assertEqual(result["protocol"]["overlap_count"], 0)
        self.assertEqual(result["protocol"]["labels_used_for_fit_or_threshold"], False)
        self.assertGreater(result["algorithms"]["multiscale_predictor"]["run_metrics"]["f1"], 0.0)
        self.assertEqual(len(result["unlabeled_run_scores"]), 1)


if __name__ == "__main__":
    unittest.main()
