# PyTorch 실행 장치를 환경 변수와 안전한 CUDA 감지로 결정하는 유틸리티
import os
import subprocess
import sys


def _cuda_check_timeout(default=10.0):
    value = os.environ.get('A_EYE_CUDA_CHECK_TIMEOUT')
    if value is None:
        return default
    try:
        return max(0.1, float(value))
    except ValueError:
        return default


def _cuda_available_via_subprocess(timeout_sec):
    code = "import torch\nprint('1' if torch.cuda.is_available() else '0')"
    try:
        completed = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and completed.stdout.strip().endswith('1')


def resolve_torch_device(env_var='A_EYE_DEVICE'):
    requested = os.environ.get(env_var, 'auto').strip().lower()
    if requested in ('', 'default'):
        requested = 'auto'
    if requested == 'auto':
        return 'cuda' if _cuda_available_via_subprocess(_cuda_check_timeout()) else 'cpu'
    return requested
