# 시간순 데이터 분할과 저비용 이상 탐지 기준선 평가를 제공하는 모듈
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TemporalPartitions:
    train: slice
    calibration: slice
    validation: slice
    test: slice


def temporal_partitions(
    length: int,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> TemporalPartitions:
    """Split ordered rows without shuffling or dropping the test remainder."""
    fractions = (train_fraction, calibration_fraction, validation_fraction)
    if length < 0:
        raise ValueError("length must be non-negative")
    if any(fraction < 0 for fraction in fractions) or sum(fractions) >= 1.0:
        raise ValueError("fractions must be non-negative and leave a non-empty test fraction")

    train_end = int(length * train_fraction)
    calibration_end = train_end + int(length * calibration_fraction)
    validation_end = calibration_end + int(length * validation_fraction)
    return TemporalPartitions(
        train=slice(0, train_end),
        calibration=slice(train_end, calibration_end),
        validation=slice(calibration_end, validation_end),
        test=slice(validation_end, length),
    )


class RobustMADDetector:
    """Score each row by its largest train-normal robust z-score."""

    def __init__(self):
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    def fit(self, normal_values: np.ndarray) -> "RobustMADDetector":
        values = _as_2d(normal_values)
        if values.shape[0] == 0:
            raise ValueError("normal_values must contain at least one row")
        center = np.median(values, axis=0)
        mad_scale = 1.4826 * np.median(np.abs(values - center), axis=0)
        std_scale = np.std(values, axis=0)
        scale = np.where(mad_scale > 1e-12, mad_scale, std_scale)
        self.center = center
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("fit must be called before score")
        rows = _as_2d(values)
        return np.max(np.abs((rows - self.center) / self.scale), axis=1)


def calibrate_quantile_threshold(scores: np.ndarray, quantile: float = 0.995) -> float:
    """Calibrate a threshold from normal-only scores."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("scores must contain at least one finite value")
    return float(np.quantile(values, quantile))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute threshold-independent average precision with tied scores grouped."""
    raw_labels = np.asarray(y_true).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if raw_labels.size == 0 or raw_labels.size != values.size:
        raise ValueError("y_true and scores must have the same non-zero length")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    if not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels")
    labels = raw_labels.astype(np.int32)

    positive_count = int(labels.sum())
    if positive_count == 0:
        return 0.0

    order = np.argsort(-values, kind="mergesort")
    sorted_scores = values[order]
    sorted_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
    true_positives = np.cumsum(sorted_labels)[group_ends]
    predicted_positives = group_ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / positive_count
    recall_step = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_step * precision))


def benchmark_prediction(
    anomaly_score: float,
    threshold: float,
    stateful_prediction: bool,
    sample_step: int,
) -> int:
    """Use runtime persistence only when benchmark windows are consecutive."""
    if sample_step < 1:
        raise ValueError("sample_step must be at least 1")
    if sample_step == 1:
        return int(bool(stateful_prediction))
    return int(float(anomaly_score) > float(threshold))


def event_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """Match predicted and true intervals once and report event latency."""
    truth = np.asarray(y_true).astype(bool).reshape(-1)
    prediction = np.asarray(y_pred).astype(bool).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    true_events = _segments(truth)
    predicted_events = _segments(prediction)
    used_predictions = set()
    latencies: List[int] = []

    for true_start, true_end in true_events:
        for prediction_idx, (pred_start, pred_end) in enumerate(predicted_events):
            if prediction_idx in used_predictions:
                continue
            if pred_start < true_end and pred_end > true_start:
                used_predictions.add(prediction_idx)
                latencies.append(max(pred_start, true_start) - true_start)
                break

    matched = len(latencies)
    precision = matched / len(predicted_events) if predicted_events else 0.0
    recall = matched / len(true_events) if true_events else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    average_latency = float(np.mean(latencies)) if latencies else None
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_events": len(true_events),
        "predicted_events": len(predicted_events),
        "matched_events": matched,
        "false_alarm_events": len(predicted_events) - matched,
        "average_latency": average_latency,
    }


def fixed_threshold_report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, object]:
    """Evaluate a pre-calibrated threshold without fitting on evaluation labels."""
    truth = np.asarray(y_true).astype(int).reshape(-1)
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    if truth.shape != score_values.shape:
        raise ValueError("y_true and scores must have the same shape")
    prediction = (score_values > threshold).astype(int)

    true_positive = int(np.sum((truth == 1) & (prediction == 1)))
    false_positive = int(np.sum((truth == 0) & (prediction == 1)))
    false_negative = int(np.sum((truth == 1) & (prediction == 0)))
    true_negative = int(np.sum((truth == 0) & (prediction == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": float(threshold),
        "point": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "event": event_detection_metrics(truth, prediction),
    }


def robust_baseline_report(
    train_values: np.ndarray,
    calibration_values: np.ndarray,
    evaluation_values: np.ndarray,
    y_true: np.ndarray,
    quantile: float = 0.995,
) -> Dict[str, object]:
    """Fit on train, calibrate on normal holdout, and evaluate once."""
    train_rows = _as_2d(train_values)
    calibration_rows = _as_2d(calibration_values)
    evaluation_rows = _as_2d(evaluation_values)
    detector = RobustMADDetector().fit(train_rows)
    calibration_scores = detector.score(calibration_rows)
    threshold = calibrate_quantile_threshold(calibration_scores, quantile=quantile)
    report = fixed_threshold_report(y_true, detector.score(evaluation_rows), threshold)
    return {
        "baseline": "robust_mad",
        "quantile": float(quantile),
        "fit_rows": int(train_rows.shape[0]),
        "calibration_rows": int(calibration_rows.shape[0]),
        "evaluation_rows": int(evaluation_rows.shape[0]),
        **report,
    }


def _as_2d(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    if rows.ndim != 2:
        raise ValueError("values must be one- or two-dimensional")
    return rows


def _segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))
