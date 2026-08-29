# 시간순 분할과 저비용 기준선 평가 규약을 검증하는 테스트
import importlib
import importlib.util
import unittest

import numpy as np


class BenchmarkProtocolTest(unittest.TestCase):
    def _module(self):
        spec = importlib.util.find_spec("src.utils.benchmark_protocol")
        self.assertIsNotNone(spec, "src.utils.benchmark_protocol 모듈이 필요합니다.")
        return importlib.import_module("src.utils.benchmark_protocol")

    def test_temporal_partitions_are_ordered_and_cover_all_rows(self):
        module = self._module()

        partitions = module.temporal_partitions(
            10,
            train_fraction=0.5,
            calibration_fraction=0.2,
            validation_fraction=0.1,
        )

        self.assertEqual(partitions.train, slice(0, 5))
        self.assertEqual(partitions.calibration, slice(5, 7))
        self.assertEqual(partitions.validation, slice(7, 8))
        self.assertEqual(partitions.test, slice(8, 10))

    def test_robust_mad_detector_scores_shifted_values_higher(self):
        module = self._module()
        detector = module.RobustMADDetector().fit(
            np.array([[0.0, 0.1], [0.1, -0.1], [-0.1, 0.0], [0.05, 0.05]])
        )

        normal_score = detector.score(np.array([[0.02, 0.0]]))[0]
        anomaly_score = detector.score(np.array([[5.0, -4.0]]))[0]

        self.assertGreater(anomaly_score, normal_score)

    def test_event_metrics_match_once_and_measure_latency(self):
        module = self._module()
        y_true = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1])

        metrics = module.event_detection_metrics(y_true, y_pred)

        self.assertEqual(metrics["true_events"], 2)
        self.assertEqual(metrics["predicted_events"], 2)
        self.assertEqual(metrics["matched_events"], 1)
        self.assertEqual(metrics["false_alarm_events"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["average_latency"], 1.0)

    def test_fixed_threshold_report_keeps_point_and_event_metrics_separate(self):
        module = self._module()
        calibration_scores = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = module.calibrate_quantile_threshold(calibration_scores, quantile=0.75)
        y_true = np.array([0, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.9, 0.1])

        report = module.fixed_threshold_report(y_true, scores, threshold)

        self.assertEqual(report["threshold"], threshold)
        self.assertEqual(report["point"]["true_positive"], 1)
        self.assertEqual(report["event"]["matched_events"], 1)

    def test_sparse_benchmark_prediction_uses_stateless_score_threshold(self):
        module = self._module()

        prediction = module.benchmark_prediction(
            anomaly_score=1.0,
            threshold=2.0,
            stateful_prediction=True,
            sample_step=10,
        )

        self.assertEqual(prediction, 0)

    def test_consecutive_benchmark_prediction_keeps_runtime_persistence(self):
        module = self._module()

        prediction = module.benchmark_prediction(
            anomaly_score=3.0,
            threshold=2.0,
            stateful_prediction=False,
            sample_step=1,
        )

        self.assertEqual(prediction, 0)

    def test_robust_baseline_report_fits_and_calibrates_before_evaluation(self):
        module = self._module()
        train = np.array([[0.0], [0.1], [-0.1], [0.05]])
        calibration = np.array([[0.0], [0.2], [-0.2], [0.1]])
        evaluation = np.array([[0.0], [4.0], [5.0], [0.1]])
        labels = np.array([0, 1, 1, 0])

        reporter = getattr(module, "robust_baseline_report", None)
        self.assertTrue(callable(reporter), "robust_baseline_report가 필요합니다.")
        report = reporter(
            train,
            calibration,
            evaluation,
            labels,
            quantile=0.75,
        )

        self.assertEqual(report["baseline"], "robust_mad")
        self.assertEqual(report["fit_rows"], 4)
        self.assertEqual(report["calibration_rows"], 4)
        self.assertEqual(report["evaluation_rows"], 4)
        self.assertEqual(report["point"]["true_positive"], 2)

    def test_average_precision_uses_score_ranking_without_test_threshold(self):
        module = self._module()
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.8, 0.7])

        result = module.average_precision(labels, scores)

        self.assertAlmostEqual(result, (1.0 + 2.0 / 3.0) / 2.0)

    def test_average_precision_rejects_fractional_labels(self):
        module = self._module()

        with self.assertRaises(ValueError):
            module.average_precision(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
