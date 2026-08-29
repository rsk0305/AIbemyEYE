# 상관된 이상 점수를 강건한 Mahalanobis 거리로 결합하는 도구
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from src.utils.benchmark_protocol import (
    average_precision,
    calibrate_quantile_threshold,
    fixed_threshold_report,
)
from src.utils.evaluation import binary_detection_metrics
from src.utils.hybrid_scoring import evaluate_normal_only_score_fusion


class RobustScoreMahalanobisFusion:
    """Model joint upper-tail score departures after robust marginal scaling."""

    def __init__(
        self,
        quantile: float = 0.995,
        shrinkage: float = 0.1,
        ridge: float = 1e-6,
    ):
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between zero and one")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        self.quantile = float(quantile)
        self.shrinkage = float(shrinkage)
        self.ridge = float(ridge)
        self.component_names: Tuple[str, ...] = ()
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.precision: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.fit_summary: Dict[str, object] = {}

    def fit(
        self,
        calibration_scores: Mapping[str, np.ndarray],
        validation_scores: Mapping[str, np.ndarray],
    ) -> "RobustScoreMahalanobisFusion":
        calibration, names = _score_matrix(calibration_scores)
        validation, validation_names = _score_matrix(validation_scores)
        if names != validation_names:
            raise ValueError("calibration and validation components must match")
        if calibration.shape[0] < 2:
            raise ValueError("calibration requires at least two score rows")

        transformed = _signed_log1p(calibration)
        center = np.median(transformed, axis=0)
        mad = 1.4826 * np.median(np.abs(transformed - center), axis=0)
        standard_deviation = np.std(transformed, axis=0)
        scale = np.where(mad > 1e-12, mad, standard_deviation)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = np.maximum((transformed - center) / scale, 0.0)
        covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            (1.0 - self.shrinkage) * covariance
            + self.shrinkage * diagonal
            + self.ridge * np.eye(len(names))
        )

        self.component_names = names
        self.center = center
        self.scale = scale
        self.precision = np.linalg.pinv(covariance, hermitian=True)
        validation_scores_combined = self.score(
            {name: validation[:, index] for index, name in enumerate(names)}
        )
        self.threshold = calibrate_quantile_threshold(
            validation_scores_combined,
            self.quantile,
        )
        self.fit_summary = {
            "method": "robust_upper_tail_log_score_mahalanobis",
            "components": list(names),
            "calibration_samples": int(calibration.shape[0]),
            "validation_samples": int(validation.shape[0]),
            "threshold_quantile": self.quantile,
            "threshold": float(self.threshold),
            "shrinkage": self.shrinkage,
            "ridge": self.ridge,
            "center": center.tolist(),
            "scale": scale.tolist(),
            "covariance": covariance.tolist(),
        }
        return self

    def score(self, component_scores: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.center is None or self.scale is None or self.precision is None:
            raise RuntimeError("fit must establish calibration statistics before scoring")
        values, names = _score_matrix(component_scores)
        if names != self.component_names:
            raise ValueError("component_scores must match the fitted components")
        standardized = np.maximum(
            (_signed_log1p(values) - self.center) / self.scale,
            0.0,
        )
        squared_distance = np.einsum(
            "bi,ij,bj->b",
            standardized,
            self.precision,
            standardized,
        )
        return np.maximum(squared_distance, 0.0)

    def predict(self, component_scores: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("fit must be called before prediction")
        return (self.score(component_scores) > self.threshold).astype(np.int32)


def evaluate_robust_score_fusion(
    calibration_scores: Mapping[str, np.ndarray],
    validation_scores: Mapping[str, np.ndarray],
    evaluation_scores: Mapping[str, np.ndarray],
    evaluation_labels: np.ndarray,
    quantile: float = 0.995,
    shrinkage: float = 0.1,
    ridge: float = 1e-6,
) -> Dict[str, object]:
    """Evaluate robust joint-score distance with the shared component protocol."""
    base_report = evaluate_normal_only_score_fusion(
        calibration_scores,
        validation_scores,
        evaluation_scores,
        evaluation_labels,
        quantile=quantile,
    )
    raw_labels = np.asarray(evaluation_labels).reshape(-1)
    labels = raw_labels.astype(np.int32)
    fusion = RobustScoreMahalanobisFusion(
        quantile=quantile,
        shrinkage=shrinkage,
        ridge=ridge,
    ).fit(calibration_scores, validation_scores)
    validation_combined = fusion.score(validation_scores)
    evaluation_combined = fusion.score(evaluation_scores)
    if evaluation_combined.size != labels.size:
        raise ValueError("evaluation scores must align one-to-one with labels")
    threshold_report = fixed_threshold_report(
        labels,
        evaluation_combined,
        float(fusion.threshold),
    )
    predictions = (evaluation_combined > float(fusion.threshold)).astype(np.int32)
    base_report["fusion"] = {
        **threshold_report,
        "method": fusion.fit_summary["method"],
        "point_adjusted": binary_detection_metrics(
            labels,
            _point_adjust(predictions, labels),
        ),
        "ranking": {
            "average_precision": average_precision(labels, evaluation_combined),
        },
        "score_summary": _score_summary(evaluation_combined),
        "validation_false_positive_rate": float(
            np.mean(validation_combined > float(fusion.threshold))
        ),
    }
    base_report["protocol"] = {
        **fusion.fit_summary,
        "normal_only_calibration": True,
        "normal_only_validation_threshold": True,
        "test_labels_used_for_threshold": False,
        "evaluation_samples": int(labels.size),
    }
    return base_report


def _score_matrix(scores: Mapping[str, np.ndarray]):
    names = tuple(sorted(scores))
    if not names:
        raise ValueError("scores must contain at least one component")
    columns = []
    for name in names:
        values = np.asarray(scores[name], dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"scores[{name}] must contain finite values")
        columns.append(values)
    lengths = {column.size for column in columns}
    if len(lengths) != 1:
        raise ValueError("score components must have the same sample count")
    return np.column_stack(columns), names


def _signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def _score_summary(scores):
    return {
        "minimum": float(np.min(scores)),
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "maximum": float(np.max(scores)),
    }


def _point_adjust(predictions, labels):
    adjusted = predictions.copy()
    padded = np.pad(labels.astype(bool), (1, 1))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        if np.any(adjusted[start:end]):
            adjusted[start:end] = 1
    return adjusted
