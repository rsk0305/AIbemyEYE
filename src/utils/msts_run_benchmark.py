# MSTS 시험 실행 단위의 정상 학습과 약한 시나리오 레이블 평가를 수행하는 도구
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from src.models.multiscale_forecaster import MultiScaleProbabilisticForecaster
from src.utils.benchmark_protocol import (
    RobustMADDetector,
    average_precision,
    calibrate_quantile_threshold,
)
from src.utils.evaluation import binary_detection_metrics
from src.utils.predictive_benchmark import rolling_windows


def run_msts_run_benchmark(
    runs: Mapping[str, np.ndarray],
    train_run_ids: Sequence[str],
    calibration_run_ids: Sequence[str],
    labeled_test_labels: Mapping[str, int],
    unlabeled_run_ids: Sequence[str] = (),
    window_size: int = 65,
    quantile: float = 0.995,
    min_consecutive_alarms: int = 3,
) -> Dict[str, object]:
    """Fit only normal train runs and evaluate scenario labels at run granularity."""
    if window_size < 3:
        raise ValueError("window_size must be at least three")
    if min_consecutive_alarms < 1:
        raise ValueError("min_consecutive_alarms must be positive")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")

    train_ids = tuple(train_run_ids)
    calibration_ids = tuple(calibration_run_ids)
    test_ids = tuple(labeled_test_labels)
    unknown_ids = tuple(unlabeled_run_ids)
    _validate_partition(runs, train_ids, calibration_ids, test_ids, unknown_ids)
    train_values = [_finite_sequence(runs[run_id], run_id) for run_id in train_ids]
    calibration_values = [
        _finite_sequence(runs[run_id], run_id) for run_id in calibration_ids
    ]
    center, scale = _fit_normalizer(train_values)
    normalized = {
        run_id: (_finite_sequence(values, run_id) - center) / scale
        for run_id, values in runs.items()
    }
    scales = tuple(scale_value for scale_value in (4, 16, 64) if scale_value < window_size)
    if not scales:
        scales = (window_size - 1,)
    train_windows = _windows_for(normalized, train_ids, window_size)
    calibration_windows = _windows_for(normalized, calibration_ids, window_size)

    predictor = MultiScaleProbabilisticForecaster(scales=scales).fit(
        train_windows,
        augment_targets=True,
    )
    predictor_threshold = calibrate_quantile_threshold(
        predictor.score_windows(calibration_windows),
        quantile,
    )
    mad = RobustMADDetector().fit(np.concatenate([normalized[run_id] for run_id in train_ids]))
    mad_threshold = calibrate_quantile_threshold(
        mad.score(np.concatenate([normalized[run_id] for run_id in calibration_ids])),
        quantile,
    )

    test_labels = np.asarray([labeled_test_labels[run_id] for run_id in test_ids], dtype=int)
    if not np.isin(test_labels, [0, 1]).all():
        raise ValueError("labeled_test_labels must contain only zero or one")
    algorithms = {
        "multiscale_predictor": _evaluate_algorithm(
            test_ids,
            test_labels,
            normalized,
            lambda sequence: predictor.score_windows(
                _run_windows(sequence, window_size)
            ),
            predictor_threshold,
            min_consecutive_alarms,
        ),
        "robust_mad": _evaluate_algorithm(
            test_ids,
            test_labels,
            normalized,
            mad.score,
            mad_threshold,
            min_consecutive_alarms,
        ),
    }
    unlabeled_run_scores = {
        run_id: {
            "multiscale_predictor": _run_score(
                predictor.score_windows(_run_windows(normalized[run_id], window_size)),
                predictor_threshold,
                min_consecutive_alarms,
            ),
            "robust_mad": _run_score(
                mad.score(normalized[run_id]),
                mad_threshold,
                min_consecutive_alarms,
            ),
        }
        for run_id in unknown_ids
        if len(normalized[run_id]) >= window_size
    }
    return {
        "protocol": {
            "evaluation_grain": "run_level_weak_scenario_label",
            "train_run_ids": list(train_ids),
            "calibration_run_ids": list(calibration_ids),
            "labeled_test_run_ids": list(test_ids),
            "unlabeled_run_ids": list(unknown_ids),
            "overlap_count": 0,
            "labels_used_for_fit_or_threshold": False,
            "labels_used_for_metrics_only": True,
            "window_size_seconds": window_size,
            "scales_seconds": list(scales),
            "score_quantile": quantile,
            "min_consecutive_alarms": min_consecutive_alarms,
            "normalization_fit_runs": list(train_ids),
        },
        "normalization": {
            "center": center.tolist(),
            "scale": scale.tolist(),
        },
        "algorithms": algorithms,
        "unlabeled_run_scores": unlabeled_run_scores,
    }


def _evaluate_algorithm(
    run_ids,
    labels,
    normalized_runs,
    score_function,
    threshold,
    min_consecutive_alarms,
):
    reports = {}
    confidences = []
    predictions = []
    for run_id in run_ids:
        report = _run_score(
            score_function(normalized_runs[run_id]),
            threshold,
            min_consecutive_alarms,
        )
        reports[run_id] = report
        confidences.append(report["p99_score"])
        predictions.append(report["predicted_anomalous"])
    return {
        "threshold": float(threshold),
        "run_reports": reports,
        "run_metrics": binary_detection_metrics(labels, np.asarray(predictions, dtype=int)),
        "run_average_precision": average_precision(labels, np.asarray(confidences)),
    }


def _run_score(scores, threshold, min_consecutive_alarms):
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("run must produce at least one anomaly score")
    alarms = values > threshold
    return {
        "score_count": int(values.size),
        "max_score": float(np.max(values)),
        "p99_score": float(np.quantile(values, 0.99)),
        "alarm_count": int(np.sum(alarms)),
        "alarm_fraction": float(np.mean(alarms)),
        "max_consecutive_alarms": _max_consecutive(alarms),
        "predicted_anomalous": int(_max_consecutive(alarms) >= min_consecutive_alarms),
    }


def _max_consecutive(values):
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _fit_normalizer(sequences):
    values = np.concatenate(sequences, axis=0)
    center = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0), 1e-6)
    return center, scale


def _windows_for(runs, run_ids, window_size):
    windows = [_run_windows(runs[run_id], window_size) for run_id in run_ids]
    return np.concatenate(windows, axis=0)


def _run_windows(sequence, window_size):
    if len(sequence) < window_size:
        raise ValueError("every selected run must contain one full window")
    windows, _ = rolling_windows(sequence, window_size=window_size)
    return windows


def _validate_partition(runs, train_ids, calibration_ids, test_ids, unknown_ids):
    groups = (train_ids, calibration_ids, test_ids, unknown_ids)
    all_ids = [run_id for group in groups for run_id in group]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("run partitions must be disjoint")
    if not train_ids or not calibration_ids or not test_ids:
        raise ValueError("train, calibration, and labeled test runs are required")
    missing = set(all_ids) - set(runs)
    if missing:
        raise ValueError(f"selected runs are unavailable: {sorted(missing)}")


def _finite_sequence(values, run_id):
    sequence = np.asarray(values, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[0] == 0 or sequence.shape[1] == 0:
        raise ValueError(f"{run_id} must have shape (time, channels)")
    if not np.isfinite(sequence).all():
        raise ValueError(f"{run_id} contains non-finite values")
    return sequence
