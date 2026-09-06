# NASA 채널별 정상 점수 안정성 전문가 선택을 기존 점수 캐시로 평가하는 스크립트
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.evaluate_nasa_hybrid import (
    DEVELOPMENT_HOLDOUT_CHANNELS,
    PRIMARY_CHANNELS,
    UPPER_TAIL_VALIDATION_CHANNELS,
)
from src.utils.channel_expert_selection import NormalStabilityExpertSelector
from src.utils.robust_score_fusion import RobustScoreMahalanobisFusion


SELECTION_DEVELOPMENT_CHANNELS = PRIMARY_CHANNELS + DEVELOPMENT_HOLDOUT_CHANNELS
EXPERT_NAMES = (
    "neural_reconstruction",
    "predictive_state",
    "robust_mahalanobis",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/nasa_hybrid_performance.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nasa_expert_selection_performance.json"),
    )
    return parser.parse_args()


def _choose_policy(stability_development_f1, neural_development_f1):
    if stability_development_f1 > neural_development_f1:
        return "normal_stability"
    return "neural_only"


def _cache_scores(channel_report):
    path = Path(channel_report["cache"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"score cache is missing: {path}")
    required = {
        "neural_calibration",
        "neural_validation",
        "predictive_calibration",
        "predictive_validation",
    }
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError(f"score cache has missing arrays: {path}")
        return {name: np.asarray(archive[name]) for name in required}


def _normal_expert_scores(scores, quantile):
    components = {
        "neural_reconstruction": {
            "calibration": scores["neural_calibration"],
            "validation": scores["neural_validation"],
        },
        "predictive_state": {
            "calibration": scores["predictive_calibration"],
            "validation": scores["predictive_validation"],
        },
    }
    robust = RobustScoreMahalanobisFusion(quantile=quantile).fit(
        {
            "neural_reconstruction": scores["neural_calibration"],
            "predictive_state": scores["predictive_calibration"],
        },
        {
            "neural_reconstruction": scores["neural_validation"],
            "predictive_state": scores["predictive_validation"],
        },
    )
    components["robust_mahalanobis"] = {
        "calibration": robust.score(
            {
                "neural_reconstruction": scores["neural_calibration"],
                "predictive_state": scores["predictive_calibration"],
            }
        ),
        "validation": robust.score(
            {
                "neural_reconstruction": scores["neural_validation"],
                "predictive_state": scores["predictive_validation"],
            }
        ),
    }
    return components


def _expert_metrics(channel_report):
    components = channel_report["components"]
    return {
        "neural_reconstruction": components["neural_reconstruction"],
        "predictive_state": components["predictive_state"],
        "robust_mahalanobis": channel_report["fusion"],
    }


def _stability_decision(channel_report):
    quantile = float(channel_report["protocol"]["threshold_quantile"])
    selector = NormalStabilityExpertSelector(
        quantile=quantile,
        fallback_expert="neural_reconstruction",
    ).fit(_normal_expert_scores(_cache_scores(channel_report), quantile))
    metrics = _expert_metrics(channel_report)
    selected = selector.selected_expert
    return {
        "channel": channel_report["channel"],
        "role": channel_report["role"],
        "selected_expert": selected,
        "selector": selector.fit_summary,
        "metrics": copy.deepcopy(metrics[selected]),
    }


def _neural_decision(channel_report):
    return {
        "channel": channel_report["channel"],
        "role": channel_report["role"],
        "selected_expert": "neural_reconstruction",
        "selector": {
            "method": "fixed_neural_baseline",
            "labels_used": False,
            "selected_expert": "neural_reconstruction",
        },
        "metrics": copy.deepcopy(_expert_metrics(channel_report)["neural_reconstruction"]),
    }


def _summary(decisions, channels):
    selected = [decisions[channel] for channel in channels if channel in decisions]
    if not selected:
        return None
    return {
        "channels": [decision["channel"] for decision in selected],
        "average_precision": float(
            np.mean([decision["metrics"]["ranking"]["average_precision"] for decision in selected])
        ),
        "average_point_f1": float(
            np.mean([decision["metrics"]["point"]["f1"] for decision in selected])
        ),
        "average_point_adjusted_f1": float(
            np.mean([decision["metrics"]["point_adjusted"]["f1"] for decision in selected])
        ),
        "expert_counts": {
            expert: sum(decision["selected_expert"] == expert for decision in selected)
            for expert in EXPERT_NAMES
        },
    }


def _policy_result(channel_reports, policy):
    decision_factory = (
        _stability_decision if policy == "normal_stability" else _neural_decision
    )
    return {channel: decision_factory(report) for channel, report in channel_reports.items()}


def evaluate(channel_reports):
    stability = _policy_result(channel_reports, "normal_stability")
    neural = _policy_result(channel_reports, "neural_only")
    stability_development = _summary(stability, SELECTION_DEVELOPMENT_CHANNELS)
    neural_development = _summary(neural, SELECTION_DEVELOPMENT_CHANNELS)
    selected_policy = _choose_policy(
        stability_development["average_point_f1"],
        neural_development["average_point_f1"],
    )
    active = stability if selected_policy == "normal_stability" else neural
    all_channels = tuple(channel_reports)
    return {
        "selection_policy": {
            "development_channels": list(SELECTION_DEVELOPMENT_CHANNELS),
            "frozen_validation_channels": list(UPPER_TAIL_VALIDATION_CHANNELS),
            "development_labels_used_to_select_between_policies": True,
            "normal_score_labels_used_for_channel_selection": False,
            "policy_candidates": ["neural_only", "normal_stability"],
            "policy_selection_rule": "choose normal_stability only for strict development Point F1 improvement",
            "selected_policy": selected_policy,
        },
        "channels": active,
        "summary": {
            "all": _summary(active, all_channels),
            "development": _summary(active, SELECTION_DEVELOPMENT_CHANNELS),
            "upper_tail_validation": _summary(active, UPPER_TAIL_VALIDATION_CHANNELS),
            "candidate_policies": {
                "normal_stability": {
                    "development": stability_development,
                    "upper_tail_validation": _summary(
                        stability,
                        UPPER_TAIL_VALIDATION_CHANNELS,
                    ),
                },
                "neural_only": {
                    "development": neural_development,
                    "upper_tail_validation": _summary(
                        neural,
                        UPPER_TAIL_VALIDATION_CHANNELS,
                    ),
                },
            },
        },
    }


def main():
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "algorithm": "normal_stability_channel_expert_selection",
        "source_report": str(args.input),
        **evaluate(source["channels"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    final = result["summary"]["upper_tail_validation"]
    print(
        f"policy={result['selection_policy']['selected_policy']}; "
        f"upper-tail Point F1={final['average_point_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
