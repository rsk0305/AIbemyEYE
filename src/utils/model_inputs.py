# 이상 탐지 모델에 전달할 비트 컨텍스트를 일관된 벡터로 구성하는 유틸리티
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def build_bit_context(
    sensor_windows: Sequence[Mapping[str, Any]],
    bit_len: int,
    context_bits: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a fixed-width bit context shared by training and inference."""
    if bit_len < 0:
        raise ValueError("bit_len must be non-negative")

    values = []
    if context_bits is not None:
        values.extend(np.asarray(context_bits, dtype=np.float32).reshape(-1).tolist())
    else:
        for sensor in sensor_windows:
            raw_signal = np.asarray(sensor.get("raw_signal", []))
            if raw_signal.size == 0:
                continue
            sensor_type = str(sensor.get("type", ""))
            if sensor_type == "discrete":
                values.append(float(raw_signal.reshape(-1)[-1]))
            elif sensor_type == "bits":
                word = int(raw_signal.reshape(-1)[-1])
                width = int(sensor.get("bit_width", min(16, bit_len)))
                values.extend(float((word >> bit_idx) & 1) for bit_idx in range(max(0, width)))

    context = np.zeros(bit_len, dtype=np.float32)
    if values and bit_len:
        copied = np.nan_to_num(np.asarray(values[:bit_len], dtype=np.float32))
        context[:len(copied)] = copied
    return context
