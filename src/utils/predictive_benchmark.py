# 경량 다중 시간축 예측 모델을 누출 없는 분할로 평가하는 벤치마크 도구
from __future__ import annotations

from time import perf_counter
from typing import Dict, Optional, Sequence

import numpy as np

from src.models.multiscale_forecaster import MultiScaleProbabilisticForecaster
from src.utils.benchmark_protocol import (
    average_precision,
    calibrate_quantile_threshold,
    fixed_threshold_report,
    temporal_partitions,
)
from src.utils.evaluation import binary_detection_metrics


def evaluate_multiscale_forecaster(
    normal_train_windows: np.ndarray,
    evaluation_windows: np.ndarray,
    evaluation_labels: np.ndarray,
    scales: Sequence[int] = (4, 16, 64),
    quantile: float = 0.995,
    ridge: float = 0.05,
    augment_targets: bool = False,
    target_step: int = 4,
    deployment_reference_windows: Optional[np.ndarray] = None,
    deployment_calibration_windows: Optional[np.ndarray] = None,
    partition_purge: int = 0,
) -> Dict[str, object]:
    """Fit on ordered normal windows, calibrate separately, then score once."""
    train = _as_windows(normal_train_windows, "normal_train_windows")
    evaluation = _as_windows(evaluation_windows, "evaluation_windows")
    raw_labels = np.asarray(evaluation_labels).reshape(-1)
    if evaluation.shape[0] != raw_labels.size:
        raise ValueError("evaluation labels must align one-to-one with windows")
    if evaluation.shape[2] != train.shape[2]:
        raise ValueError("train and evaluation windows must have the same channels")
    if not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("evaluation_labels must contain only binary labels")
    labels = raw_labels.astype(np.int32)

    if partition_purge < 0:
        raise ValueError("partition_purge must be non-negative")
    partitions = temporal_partitions(
        train.shape[0],
        train_fraction=0.7,
        calibration_fraction=0.2,
        validation_fraction=0.05,
    )
    fit_windows = train[_purge_partition_end(partitions.train, partition_purge)]
    calibration_windows = train[
        _purge_partition_end(partitions.calibration, partition_purge)
    ]
    validation_windows = train[
        _purge_partition_end(partitions.validation, partition_purge)
    ]
    if fit_windows.shape[0] < 2 or calibration_windows.shape[0] < 2:
        raise ValueError("normal training data is too short for separate fit and calibration")

    fit_started = perf_counter()
    detector = MultiScaleProbabilisticForecaster(
        scales=scales,
        ridge=ridge,
    ).fit(
        fit_windows,
        augment_targets=augment_targets,
        target_step=target_step,
    )
    fit_seconds = perf_counter() - fit_started

    deployment_inputs = (
        deployment_reference_windows is not None,
        deployment_calibration_windows is not None,
    )
    if any(deployment_inputs) and not all(deployment_inputs):
        raise ValueError("deployment reference and calibration windows must be provided together")

    adaptation_seconds = 0.0
    adaptation_window_count = 0
    reference_source = "train_holdout"
    if all(deployment_inputs):
        reference_windows = _as_windows(
            deployment_reference_windows,
            "deployment_reference_windows",
        )
        calibration_windows = _as_windows(
            deployment_calibration_windows,
            "deployment_calibration_windows",
        )
        if (
            reference_windows.shape[1:] != train.shape[1:]
            or calibration_windows.shape[1:] != train.shape[1:]
        ):
            raise ValueError("deployment windows must match train time and channel dimensions")
        adaptation_started = perf_counter()
        detector.adapt_reference(reference_windows)
        adaptation_seconds = perf_counter() - adaptation_started
        adaptation_window_count = int(reference_windows.shape[0])
        validation_windows = np.empty((0, *train.shape[1:]), dtype=np.float64)
        reference_source = "deployment_warmup"

    calibration_scores = detector.score_windows(calibration_windows)
    threshold = calibrate_quantile_threshold(calibration_scores, quantile=quantile)
    validation_scores = (
        detector.score_windows(validation_windows)
        if validation_windows.shape[0]
        else np.empty(0, dtype=np.float64)
    )

    score_started = perf_counter()
    components = detector.score_windows(evaluation, return_components=True)
    score_seconds = perf_counter() - score_started
    predictions = (components["total"] > threshold).astype(np.int32)
    adjusted_predictions = _point_adjust(predictions, labels)
    threshold_report = fixed_threshold_report(labels, components["total"], threshold)

    return {
        "algorithm": "multiscale_probabilistic_forecaster",
        "threshold": float(threshold),
        "point": threshold_report["point"],
        "point_adjusted": binary_detection_metrics(labels, adjusted_predictions),
        "event": threshold_report["event"],
        "ranking": {
            "average_precision": average_precision(labels, components["total"]),
            "forecast_average_precision": average_precision(labels, components["forecast"]),
            "state_average_precision": average_precision(labels, components["state"]),
        },
        "score_summary": {
            "minimum": float(np.min(components["total"])),
            "median": float(np.median(components["total"])),
            "mean": float(np.mean(components["total"])),
            "maximum": float(np.max(components["total"])),
        },
        "protocol": {
            "fit_windows": int(fit_windows.shape[0]),
            "calibration_windows": int(calibration_windows.shape[0]),
            "calibration_scores": int(calibration_scores.size),
            "validation_windows": int(validation_windows.shape[0]),
            "evaluation_windows": int(evaluation.shape[0]),
            "adaptation_windows": adaptation_window_count,
            "reference_source": reference_source,
            "partition_purge_windows": int(partition_purge),
            "normal_only_fit": True,
            "test_labels_used_for_threshold": False,
            "calibration_quantile": float(quantile),
            "validation_false_positive_rate": (
                float(np.mean(validation_scores > threshold))
                if validation_scores.size
                else None
            ),
        },
        "model": {
            "scales": [int(scale) for scale in detector.scales],
            "ridge": float(detector.ridge),
            "augment_targets": bool(augment_targets),
            "target_step": int(target_step),
            **detector.fit_summary,
        },
        "runtime_seconds": {
            "fit": float(fit_seconds),
            "adaptation": float(adaptation_seconds),
            "evaluation_scoring": float(score_seconds),
            "evaluation_windows_per_second": (
                float(evaluation.shape[0] / score_seconds)
                if score_seconds > 0.0
                else None
            ),
        },
    }


def combine_feature_channels(primary: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Combine primary telemetry and aligned context channels without averaging."""
    primary_values = np.asarray(primary, dtype=np.float64)
    context_values = np.asarray(context, dtype=np.float64)
    if primary_values.ndim == 1:
        primary_values = primary_values[:, None]
    if context_values.ndim == 1:
        context_values = context_values[:, None]
    if primary_values.ndim != 2 or context_values.ndim != 2:
        raise ValueError("primary and context must be one- or two-dimensional")
    if primary_values.shape[0] != context_values.shape[0]:
        raise ValueError("primary and context must have the same time length")
    combined = np.concatenate([primary_values, context_values], axis=1)
    if not np.isfinite(combined).all():
        raise ValueError("feature channels must contain only finite values")
    return combined


def disjoint_warmup_windows(
    sequence: np.ndarray,
    window_size: int,
    warmup_fraction: float,
):
    """Build adaptation and calibration windows from disjoint raw-time segments."""
    values = np.asarray(sequence, dtype=np.float64)
    if not 0.0 < warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be between 0 and 1")
    warmup_end = int(values.shape[0] * warmup_fraction)
    split_index = warmup_end // 2
    if split_index < window_size or warmup_end - split_index < window_size:
        raise ValueError("warmup segments must each contain at least one full window")
    reference, _ = rolling_windows(values[:split_index], window_size=window_size)
    calibration, _ = rolling_windows(
        values[split_index:warmup_end],
        window_size=window_size,
    )
    return reference, calibration, warmup_end


def rolling_windows(
    sequence: np.ndarray,
    window_size: int,
    step: int = 1,
):
    """Return a zero-copy rolling-window view and aligned endpoint indices."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("sequence must have shape (time, channels)")
    if window_size < 2 or window_size > values.shape[0]:
        raise ValueError("window_size must fit within the sequence")
    if step < 1:
        raise ValueError("step must be positive")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")

    view = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=window_size,
        axis=0,
    )
    windows = np.moveaxis(view, -1, 1)[::step]
    endpoints = np.arange(window_size - 1, values.shape[0], step, dtype=np.int64)
    return windows, endpoints


def _purge_partition_end(partition: slice, purge: int) -> slice:
    start = int(partition.start or 0)
    stop = int(partition.stop or start)
    return slice(start, max(start, stop - int(purge)))


def _point_adjust(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    adjusted = predictions.copy()
    for start, end in _segments(labels.astype(bool)):
        if np.any(adjusted[start:end]):
            adjusted[start:end] = 1
    return adjusted


def _segments(mask: np.ndarray):
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return zip(starts, ends)


def _as_windows(values: np.ndarray, name: str) -> np.ndarray:
    windows = np.asarray(values, dtype=np.float64)
    if windows.ndim != 3 or windows.shape[0] == 0:
        raise ValueError(f"{name} must have shape (samples, time, channels)")
    if not np.isfinite(windows).all():
        raise ValueError(f"{name} must contain only finite values")
    return windows
