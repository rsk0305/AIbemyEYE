# 다중 시간축 확률 예측기의 인과성과 이상 점수 분리도를 검증하는 테스트
import importlib
import importlib.util
import unittest

import numpy as np


class MultiScaleForecasterTest(unittest.TestCase):
    def _model_class(self):
        module_name = "src.models.multiscale_forecaster"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec, f"{module_name} 모듈이 필요합니다.")
        module = importlib.import_module(module_name)
        return module.MultiScaleProbabilisticForecaster

    @staticmethod
    def _normal_windows(count=96, length=33, channels=2, seed=7):
        rng = np.random.default_rng(seed)
        windows = []
        for _ in range(count):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amplitude = rng.normal(1.0, 0.08)
            time = np.linspace(0.0, 2.0 * np.pi, length)
            base = amplitude * np.sin(time + phase)
            values = np.stack(
                [base + 0.03 * rng.normal(size=length), 0.6 * base + 0.02 * rng.normal(size=length)],
                axis=1,
            )
            windows.append(values[:, :channels])
        return np.asarray(windows, dtype=np.float64)

    def test_prediction_is_causal_and_target_changes_only_the_score(self):
        model_class = self._model_class()
        train = self._normal_windows()
        detector = model_class(scales=(4, 8, 16), ridge=0.05).fit(train)
        samples = train[:2].copy()
        samples[1, :-1] = samples[0, :-1]
        samples[1, -1] = samples[0, -1] + 4.0

        predictions = detector.predict(samples)
        scores = detector.score_windows(samples)

        np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-12)
        self.assertGreater(scores[1], scores[0])

    def test_frozen_state_scores_higher_than_normal_state(self):
        model_class = self._model_class()
        train = self._normal_windows()
        detector = model_class(scales=(4, 8, 16), ridge=0.05).fit(train)
        normal = self._normal_windows(count=12, seed=19)
        frozen = normal.copy()
        frozen[:] = np.mean(train, axis=(0, 1), keepdims=True)

        normal_scores = detector.score_windows(normal)
        frozen_scores = detector.score_windows(frozen)

        self.assertGreater(np.median(frozen_scores), np.median(normal_scores))

    def test_score_breakdown_is_finite_and_reports_model_size(self):
        model_class = self._model_class()
        train = self._normal_windows(count=48)
        detector = model_class(scales=(4, 8, 16), ridge=0.05).fit(
            train,
            augment_targets=True,
            target_step=4,
        )

        breakdown = detector.score_windows(train[:5], return_components=True)

        self.assertEqual(breakdown["total"].shape, (5,))
        self.assertTrue(np.isfinite(breakdown["forecast"]).all())
        self.assertTrue(np.isfinite(breakdown["state"]).all())
        self.assertGreater(detector.fit_summary["training_pairs"], len(train))
        self.assertGreater(detector.fit_summary["coefficient_count"], 0)

    def test_delta_forecast_ignores_constant_level_offset_but_detects_jump(self):
        model_class = self._model_class()
        train = np.zeros((40, 33, 1), dtype=np.float64)
        detector = model_class(scales=(4, 8, 16), ridge=0.05).fit(train)
        shifted = np.full((1, 33, 1), 3.0, dtype=np.float64)
        jumped = shifted.copy()
        jumped[:, -1, :] = 5.0

        prediction = detector.predict(shifted)
        shifted_score = detector.score_windows(shifted)[0]
        jumped_score = detector.score_windows(jumped)[0]

        np.testing.assert_allclose(prediction, np.array([[3.0]]), atol=1e-12)
        self.assertEqual(shifted_score, 0.0)
        self.assertGreater(jumped_score, shifted_score)

    def test_reference_adaptation_realigns_shift_without_refitting_forecast(self):
        model_class = self._model_class()
        train = self._normal_windows(count=96)
        shifted_normal = self._normal_windows(count=24, seed=23) + 2.0
        detector = model_class(scales=(4, 8, 16), ridge=0.05).fit(train)
        coefficients_before = detector.coefficients.copy()
        score_before = np.median(detector.score_windows(shifted_normal))

        detector.adapt_reference(shifted_normal)
        score_after = np.median(detector.score_windows(shifted_normal))
        jumped = shifted_normal[:1].copy()
        jumped[:, -1, :] += 4.0

        np.testing.assert_array_equal(detector.coefficients, coefficients_before)
        self.assertLess(score_after, score_before)
        self.assertGreater(detector.score_windows(jumped)[0], score_after)


if __name__ == "__main__":
    unittest.main()
