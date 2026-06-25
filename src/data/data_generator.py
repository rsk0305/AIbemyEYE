# Stage 1 합성 데이터 생성과 저장을 담당하는 유틸리티
import os
import numpy as np
from typing import Tuple


def generate_stage1_simple_data(
    num_sensors: int = 128,
    duration_sec: float = 200.0,
    sampling_rate: int = 200,  # Hz (고정)
    num_sample_set: int = 100,  # 샘플 세트 개수 (정상 + 비정상)
    ratio_normal_samples: float = 0.95,  # 정상 샘플 비율
    data_type: str = '1word',  # 1word만
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 1 단순 데이터 생성
    
    Parameters:
    -----------
    num_sensors : int
        센서 개수 (기본: 128)
    duration_sec : float
        신호 길이 (초)
    sampling_rate : int
        샘플링 레이트 (200Hz 고정)
    num_sample_set : int
        생성할 샘플 세트 개수 (정상 + 비정상)
    ratio_normal_samples : float
        정상 샘플 비율
    data_type : str
        '1word' 만 지원
        
    Returns:
    --------
    X_data : (num_sample_set, num_sensors, time_steps)
    y_label : (num_sample_set,)  # 0: normal, 1: abnormal
    
    Notes:
    ------
    - 정상: 낮은 진폭 정현파 + 작은 노이즈
    - 비정상: 높은 진폭 또는 주파수 변화 + 큰 노이즈
    """
    if seed is not None:
        np.random.seed(seed)
    
    if data_type != '1word':
        raise ValueError("Stage 1 only supports '1word' data type")
    
    time_steps = int(duration_sec * sampling_rate)
    num_normal = int(num_sample_set * ratio_normal_samples)
    num_abnormal = num_sample_set - num_normal
    
    X_data = np.zeros((num_sample_set, num_sensors, time_steps), dtype=np.float32)
    y_label = np.zeros(num_sample_set, dtype=np.int32)
    
    # 정상 샘플 생성
    for i in range(num_normal):
        for s in range(num_sensors):
            t = np.linspace(0, duration_sec, time_steps)
            freq = np.random.uniform(0.2, 5.0)
            sig = 2000 * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
            sig += 10000 + np.random.normal(0, 50, time_steps)
            X_data[i, s] = sig
        y_label[i] = 0  # normal
    
    # 비정상 샘플 생성
    for i in range(num_normal, num_sample_set):
        for s in range(num_sensors):
            t = np.linspace(0, duration_sec, time_steps)
            # 비정상: 높은 진폭 또는 주파수 변화 + 큰 노이즈
            freq = np.random.uniform(0.2, 5.0) * np.random.uniform(1.5, 3.0)  # frequency change
            amp = 2000 * np.random.uniform(1.5, 3.0)  # amplitude change
            sig = amp * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
            sig += 10000 + np.random.normal(0, 200, time_steps)  # larger noise
            X_data[i, s] = sig
        y_label[i] = 1  # abnormal
    
    # Normalize data per sensor (standardization)
    for s in range(num_sensors):
        sensor_data = X_data[:, s, :]  # (num_sample_set, time_steps)
        mean = np.mean(sensor_data)
        std = np.std(sensor_data)
        if std > 0:
            X_data[:, s, :] = (sensor_data - mean) / std
    
    return X_data, y_label

def save_stage1_data(out_dir: str = "data/stage1", prefix: str = "stage1", **kwargs) -> Tuple[str, str]:
    """
    Stage 1 synthetic data를 normal/abnormal NPY 파일로 저장합니다.

    Returns:
    --------
    Tuple[str, str]
        저장된 normal 파일 경로와 abnormal 파일 경로.
    """
    os.makedirs(out_dir, exist_ok=True)
    X_data, y_label = generate_stage1_simple_data(**kwargs)
    normal_mask = y_label == 0
    abnormal_mask = y_label == 1
    X_normal = X_data[normal_mask]
    X_abnormal = X_data[abnormal_mask]
    path_n = os.path.join(out_dir, f'{prefix}_train_normal.npy')
    path_a = os.path.join(out_dir, f'{prefix}_train_abnormal.npy')
    np.save(path_n, X_normal)
    np.save(path_a, X_abnormal)
    return path_n, path_a
