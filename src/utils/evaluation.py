# 평가 지표와 리포트 JSON을 표준화하는 유틸리티
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def binary_detection_metrics(y_true, y_pred) -> Dict[str, float]:
    """Binary anomaly detection 지표를 동일한 이름으로 계산합니다."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        np.asarray(y_true),
        np.asarray(y_pred),
        average='binary',
        zero_division=0,
    )
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def standard_detection_report(
    dataset: str,
    point_metrics: Dict[str, float],
    point_adjusted_metrics: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """벤치마크 간 비교 가능한 공통 리포트 스키마를 만듭니다."""
    report = {
        'schema_version': 1,
        'dataset': dataset,
        'metrics': {
            'point_level': point_metrics,
            'point_adjusted': point_adjusted_metrics,
        },
    }
    if extra:
        report['extra'] = extra
    return report


def legacy_prefixed_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    """기존 JSON 키와 호환되는 prefix 기반 지표 dict를 만듭니다."""
    return {
        f'{prefix}_precision': float(metrics['precision']),
        f'{prefix}_recall': float(metrics['recall']),
        f'{prefix}_f1': float(metrics['f1']),
    }
