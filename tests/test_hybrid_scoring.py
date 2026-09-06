# 정상 보정 기반 하이브리드 점수와 배치 재구성 점수를 검증하는 테스트
import unittest
import uuid
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.hybrid_scoring import (
    NormalOnlyScoreFusion,
    batched_reconstruction_scores,
    empirical_upper_tail_p_values,
    evaluate_normal_only_score_fusion,
    load_score_cache,
    save_score_cache,
)
from src.utils.robust_score_fusion import (
    RobustScoreMahalanobisFusion,
    evaluate_robust_score_fusion,
)
from src.utils.channel_expert_selection import NormalStabilityExpertSelector
from scripts.evaluate_nasa_hybrid import _aggregate, _fit_reference_rescale, _is_cuda_device
from scripts.evaluate_nasa_expert_selection import _choose_policy
from src.data.nasa_loader import get_nasa_channel_data


class _WindowDataset(Dataset):
    def __init__(self):
        self.windows = torch.tensor(
            [
                [[0.0], [1.0], [2.0]],
                [[2.0], [4.0], [6.0]],
                [[1.0], [3.0], [5.0]],
            ],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        high = self.windows[index]
        return {
            "high": high,
            "mid": high,
            "low": high.mean().reshape(1),
            "bits": torch.zeros(2),
        }


class _OffsetDecoder(torch.nn.Module):
    def forward(self, batch):
        target = batch["high"].reshape(batch["high"].shape[0], -1)
        return {"recon": target[:, :2] + 1.0}


class _FailingDecoder(torch.nn.Module):
    def forward(self, batch):
        raise RuntimeError("expected test failure")


class HybridScoringTest(unittest.TestCase):
    @staticmethod
    def _aggregate_report(channel, hybrid_f1, existing_f1=None):
        metric = {
            "ranking": {"average_precision": hybrid_f1},
            "point": {"f1": hybrid_f1},
            "point_adjusted": {"f1": hybrid_f1},
        }
        report = {
            "channel": channel,
            "fusion": metric,
            "components": {
                "neural_reconstruction": metric,
                "predictive_state": metric,
            },
        }
        if existing_f1 is not None:
            report["existing_ada_comparison"] = {"point_f1": existing_f1}
        return report

    def test_existing_ada_delta_uses_only_comparable_channels(self):
        reports = {
            "A-1": self._aggregate_report("A-1", hybrid_f1=0.8, existing_f1=0.5),
            "A-2": self._aggregate_report("A-2", hybrid_f1=0.0),
        }

        aggregate = _aggregate(reports, ("A-1", "A-2"))

        self.assertEqual(aggregate["hybrid"]["average_point_f1"], 0.4)
        self.assertEqual(aggregate["hybrid_comparable_average_point_f1"], 0.8)
        self.assertEqual(aggregate["existing_ada_average_point_f1"], 0.5)
        self.assertAlmostEqual(aggregate["hybrid_minus_existing_ada_point_f1"], 0.3)

    def test_fit_reference_rescale_does_not_use_later_normal_rows(self):
        train = np.array([10.0, 20.0, 30.0, 100.0])
        test = np.array([15.0, 25.0])

        train_scaled, test_scaled, summary = _fit_reference_rescale(
            train,
            test,
            fit_fraction=0.5,
        )

        np.testing.assert_allclose(train_scaled, np.array([0.0, 1.0, 1.0, 1.0]))
        np.testing.assert_allclose(test_scaled, np.array([0.5, 1.0]))
        self.assertEqual(summary["reference_rows"], 2)

    def test_nasa_loader_normalizes_from_requested_fit_fraction(self):
        train = np.column_stack(
            [
                np.array([10.0, 20.0, 30.0, 100.0]),
                np.zeros((4, 2)),
            ]
        )
        test = np.column_stack(
            [
                np.array([15.0, 25.0, 5.0]),
                np.zeros((3, 2)),
            ]
        )
        metadata = {
            "A-1": {
                "anomaly_sequences": [],
                "spacecraft": "SMAP",
                "class": "[point]",
                "num_values": 3,
            }
        }

        with (
            mock.patch("src.data.nasa_loader.os.path.exists", return_value=True),
            mock.patch("src.data.nasa_loader.np.load", side_effect=[train, test]),
            mock.patch("src.data.nasa_loader.load_nasa_metadata", return_value=metadata),
        ):
            train_dataset, test_dataset, _, _ = get_nasa_channel_data(
                "A-1",
                window_size=2,
                normalization_reference_fraction=0.5,
            )

        np.testing.assert_allclose(
            train_dataset.telemetry,
            np.array([0.0, 1.0, 1.0, 1.0]),
        )
        np.testing.assert_allclose(
            test_dataset.telemetry,
            np.array([0.5, 1.0, 0.0]),
        )
        self.assertEqual(train_dataset.normalization_summary["reference_rows"], 2)

    def test_constant_fit_reference_preserves_both_change_directions(self):
        train_scaled, test_scaled, summary = _fit_reference_rescale(
            np.array([5.0, 5.0, 4.75, 5.25]),
            np.array([4.75, 5.25]),
            fit_fraction=0.5,
        )

        np.testing.assert_allclose(
            train_scaled,
            np.array([0.5, 0.5, 0.25, 0.75]),
        )
        np.testing.assert_allclose(test_scaled, np.array([0.25, 0.75]))
        self.assertTrue(summary["constant_reference"])
        self.assertEqual(summary["constant_reference_center"], 0.5)

    def test_explicit_cuda_index_is_recognized_as_cuda_device(self):
        self.assertTrue(_is_cuda_device("cuda:0"))
        self.assertFalse(_is_cuda_device("cpu"))

    def test_empirical_tail_is_smoothed_and_conservative_for_ties(self):
        reference = np.array([1.0, 1.0, 2.0, 3.0])

        p_values = empirical_upper_tail_p_values(reference, np.array([1.0, 4.0]))

        np.testing.assert_allclose(p_values, np.array([1.0, 0.2]))

    def test_fusion_uses_only_normal_calibration_and_validation_scores(self):
        calibration = {
            "neural": np.arange(10, dtype=float),
            "predictive": np.arange(10, dtype=float) * 10.0,
        }
        validation = {
            "neural": np.array([1.0, 2.0, 3.0, 4.0]),
            "predictive": np.array([10.0, 20.0, 30.0, 40.0]),
        }
        fusion = NormalOnlyScoreFusion(quantile=0.75).fit(calibration, validation)

        scores = fusion.score(
            {
                "neural": np.array([2.0, 100.0]),
                "predictive": np.array([20.0, 1000.0]),
            }
        )

        self.assertEqual(fusion.component_names, ("neural", "predictive"))
        self.assertEqual(fusion.fit_summary["calibration_samples"], 10)
        self.assertEqual(fusion.fit_summary["validation_samples"], 4)
        self.assertGreater(scores[1], scores[0])
        self.assertTrue(np.isfinite(scores).all())

    def test_batch_reconstruction_matches_per_item_mse_and_preserves_order(self):
        dataset = _WindowDataset()
        model = _OffsetDecoder()

        scores = batched_reconstruction_scores(
            model,
            dataset,
            indices=np.array([2, 0]),
            device="cpu",
            batch_size=2,
        )

        np.testing.assert_allclose(scores, np.ones(2))

    def test_batch_reconstruction_restores_model_mode_after_failure(self):
        model = _FailingDecoder()
        model.train()

        with self.assertRaisesRegex(RuntimeError, "expected test failure"):
            batched_reconstruction_scores(
                model,
                _WindowDataset(),
                device="cpu",
                batch_size=2,
            )

        self.assertTrue(model.training)

    def test_evaluation_labels_do_not_change_scores_or_thresholds(self):
        calibration = {
            "neural": np.linspace(0.0, 1.0, 20),
            "predictive": np.linspace(0.0, 2.0, 20),
        }
        validation = {
            "neural": np.array([0.1, 0.2, 0.3, 0.4]),
            "predictive": np.array([0.2, 0.4, 0.6, 0.8]),
        }
        evaluation = {
            "neural": np.array([0.1, 0.2, 4.0, 5.0]),
            "predictive": np.array([0.2, 0.4, 8.0, 10.0]),
        }
        labels = np.array([0, 0, 1, 1])

        original = evaluate_normal_only_score_fusion(
            calibration,
            validation,
            evaluation,
            labels,
            quantile=0.75,
        )
        relabeled = evaluate_normal_only_score_fusion(
            calibration,
            validation,
            evaluation,
            labels[::-1],
            quantile=0.75,
        )

        self.assertEqual(original["fusion"]["threshold"], relabeled["fusion"]["threshold"])
        self.assertEqual(
            original["fusion"]["score_summary"],
            relabeled["fusion"]["score_summary"],
        )
        self.assertEqual(
            original["components"]["neural"]["threshold"],
            relabeled["components"]["neural"]["threshold"],
        )
        self.assertNotEqual(
            original["fusion"]["ranking"]["average_precision"],
            relabeled["fusion"]["ranking"]["average_precision"],
        )

    def test_score_cache_requires_exact_metadata(self):
        path = Path.cwd() / f".hybrid-score-cache-{uuid.uuid4().hex}.npz"
        try:
            metadata = {"channel": "A-1", "seed": 42, "step": 10}
            arrays = {"calibration": np.array([0.1, 0.2])}
            save_score_cache(path, arrays, metadata)

            loaded = load_score_cache(path, expected_metadata=metadata)
            rejected = load_score_cache(
                path,
                expected_metadata={**metadata, "step": 1},
            )

            np.testing.assert_allclose(loaded["calibration"], arrays["calibration"])
            self.assertIsNone(rejected)
        finally:
            path.unlink(missing_ok=True)

    def test_robust_score_fusion_detects_joint_departure_from_correlation(self):
        x = np.linspace(0.1, 2.0, 80)
        calibration = {
            "neural": x,
            "predictive": 2.0 * x + 0.01 * np.sin(10.0 * x),
        }
        validation_x = np.linspace(0.2, 1.9, 30)
        validation = {
            "neural": validation_x,
            "predictive": 2.0 * validation_x,
        }
        fusion = RobustScoreMahalanobisFusion(quantile=0.95).fit(
            calibration,
            validation,
        )

        scores = fusion.score(
            {
                "neural": np.array([1.0, 2.0]),
                "predictive": np.array([2.0, 0.1]),
            }
        )

        self.assertGreater(scores[1], scores[0])
        self.assertEqual(
            fusion.fit_summary["method"],
            "robust_upper_tail_log_score_mahalanobis",
        )

    def test_robust_score_fusion_ignores_lower_tail_departures(self):
        x = np.linspace(1.0, 2.0, 80)
        calibration = {"neural": x, "predictive": 2.0 * x}
        validation = {
            "neural": np.linspace(1.1, 1.9, 30),
            "predictive": np.linspace(2.2, 3.8, 30),
        }
        fusion = RobustScoreMahalanobisFusion(quantile=0.95).fit(
            calibration,
            validation,
        )

        scores = fusion.score(
            {
                "neural": np.array([0.0, 4.0]),
                "predictive": np.array([0.0, 8.0]),
            }
        )

        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[1], scores[0])

    def test_robust_fusion_threshold_is_independent_of_evaluation_labels(self):
        calibration = {
            "neural": np.linspace(0.1, 1.0, 30),
            "predictive": np.linspace(0.2, 2.0, 30),
        }
        validation = {
            "neural": np.linspace(0.2, 0.9, 12),
            "predictive": np.linspace(0.4, 1.8, 12),
        }
        evaluation = {
            "neural": np.array([0.2, 0.3, 2.0, 3.0]),
            "predictive": np.array([0.4, 0.6, 4.0, 0.1]),
        }
        labels = np.array([0, 0, 1, 1])

        original = evaluate_robust_score_fusion(
            calibration,
            validation,
            evaluation,
            labels,
            quantile=0.95,
        )
        relabeled = evaluate_robust_score_fusion(
            calibration,
            validation,
            evaluation,
            labels[::-1],
            quantile=0.95,
        )

        self.assertEqual(original["fusion"]["threshold"], relabeled["fusion"]["threshold"])
        self.assertEqual(
            original["fusion"]["score_summary"],
            relabeled["fusion"]["score_summary"],
        )

    def test_normal_stability_selector_prefers_expert_with_stable_normal_tail(self):
        selector = NormalStabilityExpertSelector(
            quantile=0.8,
            fallback_expert="neural_reconstruction",
        ).fit(
            {
                "neural_reconstruction": {
                    "calibration": np.arange(10, dtype=float),
                    "validation": np.arange(10, 20, dtype=float),
                },
                "predictive_state": {
                    "calibration": np.arange(10, dtype=float),
                    "validation": np.arange(10, dtype=float),
                },
            }
        )

        self.assertEqual(selector.selected_expert, "predictive_state")
        self.assertEqual(selector.fit_summary["labels_used"], False)
        self.assertLess(
            selector.fit_summary["experts"]["predictive_state"]["tail_error"],
            selector.fit_summary["experts"]["neural_reconstruction"]["tail_error"],
        )

    def test_normal_stability_selector_keeps_fallback_for_tied_experts(self):
        scores = {
            "neural_reconstruction": {
                "calibration": np.arange(10, dtype=float),
                "validation": np.arange(10, dtype=float),
            },
            "predictive_state": {
                "calibration": np.arange(10, dtype=float),
                "validation": np.arange(10, dtype=float),
            },
        }

        selector = NormalStabilityExpertSelector(
            quantile=0.8,
            fallback_expert="neural_reconstruction",
        ).fit(scores)

        self.assertEqual(selector.selected_expert, "neural_reconstruction")

    def test_policy_choice_requires_strict_development_improvement(self):
        self.assertEqual(_choose_policy(0.30, 0.30), "neural_only")
        self.assertEqual(_choose_policy(0.31, 0.30), "normal_stability")

    def test_expert_selection_script_imports_when_run_directly(self):
        result = subprocess.run(
            [sys.executable, "scripts/evaluate_nasa_expert_selection.py", "--help"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
