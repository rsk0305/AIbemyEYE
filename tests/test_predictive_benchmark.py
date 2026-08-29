# 정상 학습·보정 분리와 테스트 라벨 비누출을 검증하는 예측 벤치마크 테스트
import importlib
import importlib.util
import unittest

import numpy as np


class PredictiveBenchmarkTest(unittest.TestCase):
    def _module(self):
        module_name = "src.utils.predictive_benchmark"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec, f"{module_name} 모듈이 필요합니다.")
        return importlib.import_module(module_name)

    def _evaluate(self):
        return self._module().evaluate_multiscale_forecaster

    @staticmethod
    def _windows(count, seed, offset=0.0):
        rng = np.random.default_rng(seed)
        time = np.linspace(0.0, 2.0 * np.pi, 33)
        windows = []
        for _ in range(count):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            signal = np.sin(time + phase) + offset
            windows.append((signal + 0.03 * rng.normal(size=time.size))[:, None])
        return np.asarray(windows)

    def test_report_separates_fit_calibration_and_test_labels(self):
        evaluate = self._evaluate()
        train = self._windows(100, seed=1)
        evaluation = np.concatenate(
            [self._windows(20, seed=2), self._windows(20, seed=3, offset=3.0)],
            axis=0,
        )
        labels = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]

        report = evaluate(
            train,
            evaluation,
            labels,
            scales=(4, 8, 16),
            quantile=0.95,
        )

        self.assertEqual(report["protocol"]["fit_windows"], 70)
        self.assertEqual(report["protocol"]["calibration_windows"], 20)
        self.assertEqual(report["protocol"]["evaluation_windows"], 40)
        self.assertGreater(report["ranking"]["average_precision"], 0.95)
        self.assertGreater(report["point"]["true_positive"], 0)

    def test_changing_test_labels_does_not_change_threshold_or_scores(self):
        evaluate = self._evaluate()
        train = self._windows(80, seed=4)
        evaluation = np.concatenate(
            [self._windows(10, seed=5), self._windows(10, seed=6, offset=2.0)],
            axis=0,
        )
        labels = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]

        original = evaluate(
            train,
            evaluation,
            labels,
            scales=(4, 8, 16),
            quantile=0.95,
        )
        relabeled = evaluate(
            train,
            evaluation,
            labels[::-1],
            scales=(4, 8, 16),
            quantile=0.95,
        )

        self.assertEqual(original["threshold"], relabeled["threshold"])
        self.assertEqual(original["score_summary"], relabeled["score_summary"])
        self.assertNotEqual(
            original["ranking"]["average_precision"],
            relabeled["ranking"]["average_precision"],
        )

    def test_rolling_windows_return_matching_endpoint_indices(self):
        module = self._module()
        sequence = np.arange(7, dtype=float)

        windows, endpoints = module.rolling_windows(sequence, window_size=3, step=2)

        np.testing.assert_array_equal(
            windows[:, :, 0],
            np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]),
        )
        np.testing.assert_array_equal(endpoints, np.array([2, 4, 6]))

    def test_augmented_fit_keeps_calibration_at_evaluation_grain(self):
        evaluate = self._evaluate()
        train = self._windows(40, seed=8)
        evaluation = self._windows(8, seed=9)

        report = evaluate(
            train,
            evaluation,
            np.zeros(8, dtype=int),
            scales=(4, 8, 16),
            quantile=0.95,
            augment_targets=True,
            target_step=4,
        )

        self.assertEqual(
            report["protocol"]["calibration_scores"],
            report["protocol"]["calibration_windows"],
        )

    def test_feature_channels_keep_primary_and_context_values(self):
        module = self._module()
        primary = np.array([0.1, 0.2, 0.3])
        context = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

        combined = module.combine_feature_channels(primary, context)

        np.testing.assert_array_equal(
            combined,
            np.array(
                [
                    [0.1, 0.0, 1.0],
                    [0.2, 1.0, 0.0],
                    [0.3, 1.0, 1.0],
                ]
            ),
        )

    def test_deployment_reference_is_declared_and_used_for_calibration(self):
        evaluate = self._evaluate()
        train = self._windows(80, seed=10)
        deployment_normal = self._windows(30, seed=11) + 2.0
        evaluation = self._windows(10, seed=12) + 2.0

        report = evaluate(
            train,
            evaluation,
            np.zeros(10, dtype=int),
            scales=(4, 8, 16),
            quantile=0.95,
            deployment_reference_windows=deployment_normal[:15],
            deployment_calibration_windows=deployment_normal[15:],
        )

        self.assertEqual(report["protocol"]["reference_source"], "deployment_warmup")
        self.assertEqual(report["protocol"]["adaptation_windows"], 15)
        self.assertEqual(report["protocol"]["calibration_windows"], 15)

    def test_partition_purge_removes_overlapping_boundary_windows(self):
        evaluate = self._evaluate()
        train = self._windows(100, seed=13)
        evaluation = self._windows(8, seed=14)

        report = evaluate(
            train,
            evaluation,
            np.zeros(8, dtype=int),
            scales=(4, 8, 16),
            quantile=0.95,
            partition_purge=3,
        )

        self.assertEqual(report["protocol"]["fit_windows"], 67)
        self.assertEqual(report["protocol"]["calibration_windows"], 17)
        self.assertEqual(report["protocol"]["validation_windows"], 2)
        self.assertEqual(report["protocol"]["partition_purge_windows"], 3)

    def test_warmup_windows_are_built_from_disjoint_raw_segments(self):
        module = self._module()
        sequence = np.arange(100, dtype=float)

        reference, calibration, warmup_end = module.disjoint_warmup_windows(
            sequence,
            window_size=10,
            warmup_fraction=0.4,
        )

        self.assertEqual(warmup_end, 40)
        self.assertLess(np.max(reference), np.min(calibration))

    def test_benchmark_rejects_fractional_evaluation_labels(self):
        evaluate = self._evaluate()
        train = self._windows(40, seed=15)
        evaluation = self._windows(3, seed=16)

        with self.assertRaises(ValueError):
            evaluate(
                train,
                evaluation,
                np.array([0.0, 0.5, 1.0]),
                scales=(4, 8, 16),
                quantile=0.95,
            )


if __name__ == "__main__":
    unittest.main()
