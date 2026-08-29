# 다중 시간축 정상 상태와 다음 시점 분포를 학습하는 경량 이상 탐지 모델
from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np


ScoreResult = Union[np.ndarray, Dict[str, np.ndarray]]


class MultiScaleProbabilisticForecaster:
    """Normal-only ridge forecaster with a diagonal Gaussian state model."""

    def __init__(
        self,
        scales: Sequence[int] = (4, 16, 64),
        ridge: float = 0.05,
        min_scale: float = 1e-3,
    ):
        normalized_scales = tuple(sorted({int(scale) for scale in scales}))
        if not normalized_scales or normalized_scales[0] <= 0:
            raise ValueError("scales must contain positive integers")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if min_scale <= 0.0:
            raise ValueError("min_scale must be positive")

        self.scales = normalized_scales
        self.ridge = float(ridge)
        self.min_scale = float(min_scale)
        self.fit_summary: Dict[str, int] = {}
        self._is_fitted = False

    def fit(
        self,
        windows: np.ndarray,
        augment_targets: bool = False,
        target_step: int = 4,
    ) -> "MultiScaleProbabilisticForecaster":
        values = self._validate_windows(windows)
        if target_step <= 0:
            raise ValueError("target_step must be positive")

        forecast_features, targets, state_features = self._training_design(
            values,
            augment_targets=augment_targets,
            target_step=target_step,
        )
        self.channel_count = values.shape[2]
        self.feature_center = forecast_features.mean(axis=0)
        self.feature_scale = self._standard_scale(forecast_features)
        normalized_features = (forecast_features - self.feature_center) / self.feature_scale

        history_endpoints = forecast_features[:, : self.channel_count]
        target_deltas = targets - history_endpoints
        self.target_center = target_deltas.mean(axis=0)
        self.target_scale = self._standard_scale(target_deltas)
        normalized_deltas = (target_deltas - self.target_center) / self.target_scale
        self.coefficients = self._fit_ridge(normalized_features, normalized_deltas)

        fitted_targets = history_endpoints + self._decode_delta(
            normalized_features @ self.coefficients
        )
        residuals = targets - fitted_targets
        self.residual_center = np.median(residuals, axis=0)
        self.residual_scale = np.maximum(
            self._estimated_robust_scale(residuals),
            self.min_scale,
        )

        self.state_center = np.median(state_features, axis=0)
        estimated_state_scale = self._estimated_robust_scale(state_features)
        self.state_active = estimated_state_scale > self.min_scale
        self.state_scale = np.maximum(estimated_state_scale, self.min_scale)
        self.fit_summary = {
            "training_windows": int(values.shape[0]),
            "training_pairs": int(targets.shape[0]),
            "channels": int(self.channel_count),
            "feature_count": int(forecast_features.shape[1]),
            "coefficient_count": int(self.coefficients.size),
        }
        self._is_fitted = True
        return self

    def adapt_reference(
        self,
        normal_windows: np.ndarray,
    ) -> "MultiScaleProbabilisticForecaster":
        """Realign residual and state distributions without refitting forecast weights."""
        values = self._validate_windows(normal_windows)
        self._require_fitted(values)
        if values.shape[0] < 2:
            raise ValueError("normal_windows must contain at least two windows")

        residuals = values[:, -1, :] - self.predict(values)
        self.residual_center = np.median(residuals, axis=0)
        self.residual_scale = np.maximum(
            self._estimated_robust_scale(residuals),
            self.min_scale,
        )

        state_features = self._summary_features(values)
        self.state_center = np.median(state_features, axis=0)
        estimated_state_scale = self._estimated_robust_scale(state_features)
        self.state_active = estimated_state_scale > self.min_scale
        self.state_scale = np.maximum(estimated_state_scale, self.min_scale)
        self.fit_summary["adaptation_windows"] = int(values.shape[0])
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        values = self._validate_windows(windows)
        self._require_fitted(values)
        features = self._summary_features(values[:, :-1, :])
        normalized = (features - self.feature_center) / self.feature_scale
        return values[:, -2, :] + self._decode_delta(normalized @ self.coefficients)

    def score_windows(
        self,
        windows: np.ndarray,
        return_components: bool = False,
    ) -> ScoreResult:
        values = self._validate_windows(windows)
        self._require_fitted(values)
        predictions = self.predict(values)
        residual_z = (
            values[:, -1, :] - predictions - self.residual_center
        ) / self.residual_scale
        state_features = self._summary_features(values)
        state_z = (state_features - self.state_center) / self.state_scale
        state_z[:, ~self.state_active] = 0.0

        forecast_score = 0.5 * np.mean(np.square(np.clip(residual_z, -1e6, 1e6)), axis=1)
        state_score = 0.5 * np.mean(np.square(np.clip(state_z, -1e6, 1e6)), axis=1)
        total_score = forecast_score + state_score
        if return_components:
            return {
                "total": total_score,
                "forecast": forecast_score,
                "state": state_score,
            }
        return total_score

    def _training_design(
        self,
        windows: np.ndarray,
        augment_targets: bool,
        target_step: int,
    ):
        if not augment_targets:
            return (
                self._summary_features(windows[:, :-1, :]),
                windows[:, -1, :],
                self._summary_features(windows),
            )

        last_target = windows.shape[1] - 1
        target_indices = list(range(max(self.scales), windows.shape[1], target_step))
        if target_indices[-1] != last_target:
            target_indices.append(last_target)

        forecast_parts = []
        target_parts = []
        state_parts = []
        for target_index in target_indices:
            forecast_parts.append(self._summary_features(windows[:, :target_index, :]))
            target_parts.append(windows[:, target_index, :])
            state_parts.append(self._summary_features(windows[:, :target_index + 1, :]))
        return (
            np.concatenate(forecast_parts, axis=0),
            np.concatenate(target_parts, axis=0),
            np.concatenate(state_parts, axis=0),
        )

    def _summary_features(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.shape[1] < max(self.scales):
            raise ValueError("window history is shorter than the largest scale")

        features = [sequences[:, -1, :]]
        for scale in self.scales:
            tail = sequences[:, -scale:, :]
            features.extend(
                [
                    tail.mean(axis=1),
                    tail.std(axis=1),
                    (tail[:, -1, :] - tail[:, 0, :]) / float(scale),
                ]
            )
        return np.concatenate(features, axis=1)

    def _fit_ridge(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        sample_count, feature_count = features.shape
        penalty = self.ridge * sample_count
        if sample_count <= feature_count:
            gram = features @ features.T
            dual = np.linalg.solve(
                gram + penalty * np.eye(sample_count, dtype=np.float64),
                targets,
            )
            return features.T @ dual

        gram = features.T @ features
        return np.linalg.solve(
            gram + penalty * np.eye(feature_count, dtype=np.float64),
            features.T @ targets,
        )

    def _decode_delta(self, normalized_deltas: np.ndarray) -> np.ndarray:
        return normalized_deltas * self.target_scale + self.target_center

    def _standard_scale(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(np.std(values, axis=0), self.min_scale)

    def _estimated_robust_scale(self, values: np.ndarray) -> np.ndarray:
        center = np.median(values, axis=0)
        mad_scale = 1.4826 * np.median(np.abs(values - center), axis=0)
        standard_scale = np.std(values, axis=0)
        return np.where(mad_scale > self.min_scale, mad_scale, standard_scale)

    def _validate_windows(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] == 0:
            raise ValueError("windows must have shape (samples, time, channels)")
        if values.shape[1] <= max(self.scales):
            raise ValueError("windows must include history and one target after the largest scale")
        if not np.isfinite(values).all():
            raise ValueError("windows must contain only finite values")
        return values

    def _require_fitted(self, windows: np.ndarray) -> None:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before prediction")
        if windows.shape[2] != self.channel_count:
            raise ValueError("window channel count does not match the fitted model")
