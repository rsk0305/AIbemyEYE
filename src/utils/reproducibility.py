# Python과 NumPy 및 PyTorch의 벤치마크 난수 시드를 함께 설정하는 유틸리티
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed all random generators used by the local benchmark code."""
    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    return normalized_seed
