# 정상 구간 안정성을 기준으로 채널별 이상 점수 전문가를 선택하는 도구
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from src.utils.benchmark_protocol import calibrate_quantile_threshold


class NormalStabilityExpertSelector:
    """Choose one score expert using normal-only tail stability."""

    def __init__(self, quantile: float = 0.995, fallback_expert: str = "neural_reconstruction"):
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be between zero and one")
        self.quantile = float(quantile)
        self.fallback_expert = fallback_expert
        self.selected_expert = None
        self.fit_summary: Dict[str, object] = {}

    def fit(
        self,
        expert_scores: Mapping[str, Mapping[str, np.ndarray]],
    ) -> "NormalStabilityExpertSelector":
        if self.fallback_expert not in expert_scores:
            raise ValueError("fallback_expert must be present in expert_scores")

        target_rate = 1.0 - self.quantile
        summaries = {}
        for name, score_sets in expert_scores.items():
            if set(score_sets) != {"calibration", "validation"}:
                raise ValueError("each expert must provide calibration and validation scores")
            calibration = _finite_vector(score_sets["calibration"], name, "calibration")
            validation = _finite_vector(score_sets["validation"], name, "validation")
            threshold = float(calibrate_quantile_threshold(calibration, self.quantile))
            validation_tail_rate = float(np.mean(validation > threshold))
            summaries[name] = {
                "threshold": threshold,
                "calibration_samples": int(calibration.size),
                "validation_samples": int(validation.size),
                "validation_tail_rate": validation_tail_rate,
                "tail_error": float(abs(validation_tail_rate - target_rate)),
            }

        self.selected_expert = min(
            summaries,
            key=lambda name: (
                summaries[name]["tail_error"],
                name != self.fallback_expert,
                name,
            ),
        )
        self.fit_summary = {
            "method": "normal_calibration_tail_stability",
            "labels_used": False,
            "quantile": self.quantile,
            "target_tail_rate": target_rate,
            "fallback_expert": self.fallback_expert,
            "selected_expert": self.selected_expert,
            "experts": summaries,
        }
        return self


def _finite_vector(values: np.ndarray, expert: str, split: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{expert} {split} scores must not be empty")
    if not np.isfinite(vector).all():
        raise ValueError(f"{expert} {split} scores must be finite")
    return vector
