# 정상 보정 점수 결합과 신경망 배치 점수 캐시를 제공하는 도구
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.benchmark_protocol import (
    average_precision,
    calibrate_quantile_threshold,
    fixed_threshold_report,
)
from src.utils.evaluation import binary_detection_metrics


def empirical_upper_tail_p_values(
    reference_scores: np.ndarray,
    query_scores: np.ndarray,
) -> np.ndarray:
    """Map scores to finite-sample upper-tail probabilities from normal data."""
    reference = _finite_vector(reference_scores, "reference_scores")
    query = _finite_vector(query_scores, "query_scores", allow_empty=True)
    ordered = np.sort(reference)
    first_equal = np.searchsorted(ordered, query, side="left")
    exceedance_count = ordered.size - first_equal
    return (exceedance_count + 1.0) / (ordered.size + 1.0)


class NormalOnlyScoreFusion:
    """Calibrate heterogeneous anomaly scores without evaluation labels."""

    def __init__(self, quantile: float = 0.995):
        self.quantile = float(quantile)
        self.component_names: Tuple[str, ...] = ()
        self.reference_scores: Dict[str, np.ndarray] = {}
        self.threshold: Optional[float] = None
        self.fit_summary: Dict[str, object] = {}

    def fit(
        self,
        calibration_scores: Mapping[str, np.ndarray],
        validation_scores: Mapping[str, np.ndarray],
    ) -> "NormalOnlyScoreFusion":
        names = tuple(sorted(calibration_scores))
        if not names:
            raise ValueError("calibration_scores must contain at least one component")
        if set(validation_scores) != set(names):
            raise ValueError("calibration and validation components must match")

        references = {
            name: _finite_vector(calibration_scores[name], f"calibration_scores[{name}]")
            for name in names
        }
        calibration_lengths = {values.size for values in references.values()}
        if len(calibration_lengths) != 1:
            raise ValueError("calibration components must have the same sample count")

        validation = {
            name: _finite_vector(validation_scores[name], f"validation_scores[{name}]")
            for name in names
        }
        validation_lengths = {values.size for values in validation.values()}
        if len(validation_lengths) != 1:
            raise ValueError("validation components must have the same sample count")

        self.component_names = names
        self.reference_scores = references
        validation_fused = self.score(validation)
        self.threshold = calibrate_quantile_threshold(validation_fused, self.quantile)
        self.fit_summary = {
            "method": "empirical_upper_tail_bonferroni_min_p",
            "components": list(names),
            "calibration_samples": int(next(iter(calibration_lengths))),
            "validation_samples": int(next(iter(validation_lengths))),
            "threshold_quantile": self.quantile,
            "threshold": float(self.threshold),
        }
        return self

    def component_p_values(
        self,
        component_scores: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if not self.component_names or not self.reference_scores:
            raise RuntimeError("fit must establish calibration references before scoring")
        if set(component_scores) != set(self.component_names):
            raise ValueError("component_scores must match the fitted components")
        values = {
            name: _finite_vector(component_scores[name], f"component_scores[{name}]")
            for name in self.component_names
        }
        lengths = {scores.size for scores in values.values()}
        if len(lengths) != 1:
            raise ValueError("component scores must have the same sample count")
        return {
            name: empirical_upper_tail_p_values(self.reference_scores[name], values[name])
            for name in self.component_names
        }

    def score(self, component_scores: Mapping[str, np.ndarray]) -> np.ndarray:
        p_values = self.component_p_values(component_scores)
        stacked = np.column_stack([p_values[name] for name in self.component_names])
        combined_p = np.minimum(1.0, stacked.shape[1] * np.min(stacked, axis=1))
        return -np.log(combined_p)

    def predict(self, component_scores: Mapping[str, np.ndarray]) -> np.ndarray:
        self._require_fitted()
        return (self.score(component_scores) > float(self.threshold)).astype(np.int32)

    def _require_fitted(self) -> None:
        if not self.component_names or self.threshold is None:
            raise RuntimeError("fit must be called before scoring")


def evaluate_normal_only_score_fusion(
    calibration_scores: Mapping[str, np.ndarray],
    validation_scores: Mapping[str, np.ndarray],
    evaluation_scores: Mapping[str, np.ndarray],
    evaluation_labels: np.ndarray,
    quantile: float = 0.995,
) -> Dict[str, object]:
    """Compare each component and their fusion under one normal-only protocol."""
    raw_labels = np.asarray(evaluation_labels).reshape(-1)
    if raw_labels.size == 0 or not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("evaluation_labels must contain binary labels")
    labels = raw_labels.astype(np.int32)

    fusion = NormalOnlyScoreFusion(quantile=quantile).fit(
        calibration_scores,
        validation_scores,
    )
    if set(evaluation_scores) != set(fusion.component_names):
        raise ValueError("evaluation components must match calibration components")

    component_reports = {}
    for name in fusion.component_names:
        validation = _finite_vector(validation_scores[name], f"validation_scores[{name}]")
        evaluation = _finite_vector(evaluation_scores[name], f"evaluation_scores[{name}]")
        if evaluation.size != labels.size:
            raise ValueError("evaluation scores must align one-to-one with labels")
        threshold = calibrate_quantile_threshold(validation, quantile)
        threshold_report = fixed_threshold_report(labels, evaluation, threshold)
        predictions = (evaluation > threshold).astype(np.int32)
        component_reports[name] = {
            **threshold_report,
            "point_adjusted": binary_detection_metrics(
                labels,
                _point_adjust(predictions, labels),
            ),
            "ranking": {"average_precision": average_precision(labels, evaluation)},
            "score_summary": _score_summary(evaluation),
            "validation_false_positive_rate": float(np.mean(validation > threshold)),
        }

    fused_validation = fusion.score(validation_scores)
    fused_evaluation = fusion.score(evaluation_scores)
    if fused_evaluation.size != labels.size:
        raise ValueError("evaluation scores must align one-to-one with labels")
    fused_report = fixed_threshold_report(labels, fused_evaluation, float(fusion.threshold))
    fused_predictions = (fused_evaluation > float(fusion.threshold)).astype(np.int32)

    return {
        "fusion": {
            **fused_report,
            "method": fusion.fit_summary["method"],
            "point_adjusted": binary_detection_metrics(
                labels,
                _point_adjust(fused_predictions, labels),
            ),
            "ranking": {
                "average_precision": average_precision(labels, fused_evaluation),
            },
            "score_summary": _score_summary(fused_evaluation),
            "validation_false_positive_rate": float(
                np.mean(fused_validation > float(fusion.threshold))
            ),
        },
        "components": component_reports,
        "protocol": {
            **fusion.fit_summary,
            "normal_only_calibration": True,
            "normal_only_validation_threshold": True,
            "test_labels_used_for_threshold": False,
            "evaluation_samples": int(labels.size),
        },
    }


def batched_reconstruction_scores(
    model: torch.nn.Module,
    dataset: Dataset,
    indices: Optional[Sequence[int]] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Compute the inference reconstruction MSE for ordered dataset windows."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    selected: Dataset = dataset
    if indices is not None:
        selected_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if selected_indices.size and (
            np.min(selected_indices) < 0 or np.max(selected_indices) >= len(dataset)
        ):
            raise IndexError("indices must refer to rows in dataset")
        selected = Subset(dataset, selected_indices.tolist())
    if len(selected) == 0:
        return np.empty(0, dtype=np.float64)

    torch_device = torch.device(device)
    loader = DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    score_parts = []
    try:
        with torch.inference_mode():
            for batch in loader:
                model_batch = {
                    key: value.to(
                        torch_device,
                        non_blocking=torch_device.type == "cuda",
                    )
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                output = model(model_batch)
                reconstruction = output["recon"]
                target = model_batch["high"].reshape(model_batch["high"].shape[0], -1)
                if reconstruction.ndim != 2 or target.shape[1] < reconstruction.shape[1]:
                    raise ValueError("model reconstruction must fit within flattened high input")
                errors = torch.mean(
                    (reconstruction - target[:, : reconstruction.shape[1]]) ** 2,
                    dim=1,
                )
                score_parts.append(
                    errors.detach().cpu().numpy().astype(np.float64, copy=False)
                )
    finally:
        model.train(was_training)
    return np.concatenate(score_parts)


def save_score_cache(
    path: Union[str, Path],
    score_arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> None:
    """Persist score arrays and their exact cache identity without pickle."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(values) for name, values in score_arrays.items()}
    if "_metadata_json" in payload:
        raise ValueError("_metadata_json is a reserved cache key")
    if any(not np.isfinite(values).all() for values in payload.values()):
        raise ValueError("score cache arrays must contain only finite values")
    payload["_metadata_json"] = np.asarray(_canonical_metadata(metadata))
    np.savez_compressed(target, **payload)


def load_score_cache(
    path: Union[str, Path],
    expected_metadata: Mapping[str, object],
) -> Optional[Dict[str, np.ndarray]]:
    """Load scores only when the caller's full metadata identity matches."""
    source = Path(path)
    if not source.exists():
        return None
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = str(archive["_metadata_json"].item())
            if metadata != _canonical_metadata(expected_metadata):
                return None
            return {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "_metadata_json"
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _canonical_metadata(metadata: Mapping[str, object]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _score_summary(scores: np.ndarray) -> Dict[str, float]:
    return {
        "minimum": float(np.min(scores)),
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "maximum": float(np.max(scores)),
    }


def _point_adjust(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    adjusted = predictions.copy()
    padded = np.pad(labels.astype(bool), (1, 1))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        if np.any(adjusted[start:end]):
            adjusted[start:end] = 1
    return adjusted


def _finite_vector(
    values: np.ndarray,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not allow_empty and vector.size == 0:
        raise ValueError(f"{name} must contain at least one score")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite scores")
    return vector
