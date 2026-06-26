# SWaT CSV 벤치마크 데이터를 NumPy 배열과 Dataset으로 변환하는 로더
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SWaTDataset(Dataset):
    def __init__(self, df_values, bit_col_indices, window_size=100, step=1, window_labels=None):
        arr = np.asarray(df_values, dtype=np.float64)
        self.prebuilt_windows = arr.ndim == 3
        if self.prebuilt_windows:
            self.windows = arr
            self.values = arr.reshape(-1, arr.shape[-1])
            self.length = arr.shape[0]
        else:
            self.windows = None
            self.values = arr  # Numpy array (T, D)
            self.length = max(0, (len(arr) - window_size) // step + 1)
        self.bit_col_indices = bit_col_indices
        self.window_size = window_size
        self.step = step
        self.window_labels = window_labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.prebuilt_windows:
            window_data = self.windows[idx]
        else:
            start = idx * self.step
            end = start + self.window_size
            window_data = self.values[start:end]  # (W, D)
        x_analog = window_data
        last_step_data = window_data[-1]
        x_bits = last_step_data[self.bit_col_indices]
        return {
            'high': torch.tensor(x_analog, dtype=torch.float32),
            'mid': torch.tensor(x_analog, dtype=torch.float32),
            'low': torch.tensor(x_analog.mean(axis=0), dtype=torch.float32),
            'bits': torch.tensor(x_bits, dtype=torch.float32),
        }


def _read_header(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return [col.strip() for col in next(csv.reader(f))]


def _feature_columns(header):
    excluded = {'Timestamp', 'Normal/Attack'}
    return [idx for idx, col in enumerate(header) if col not in excluded]


def _parse_feature_row(row, usecols):
    values = []
    for idx in usecols:
        try:
            values.append(float(row[idx].strip()))
        except (IndexError, ValueError):
            values.append(np.nan)
    return values


def _load_sampled_windows(path, usecols, window_size, window_stride, label_col_idx=None):
    if window_stride < window_size:
        raise ValueError('window_stride must be greater than or equal to window_size')

    windows = []
    labels = []
    active_window = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row_idx, row in enumerate(reader):
            cycle = row_idx % window_stride
            if cycle == 0:
                active_window = []
            if cycle >= window_size:
                continue

            active_window.append(_parse_feature_row(row, usecols))
            if cycle == window_size - 1 and len(active_window) == window_size:
                windows.append(active_window)
                if label_col_idx is not None:
                    value = row[label_col_idx].strip().lower() if len(row) > label_col_idx else ''
                    labels.append(1 if value == 'attack' else 0)
                active_window = []

    if not windows:
        empty = np.empty((0, window_size, len(usecols)), dtype=np.float64)
        empty_labels = np.empty((0,), dtype=np.int32)
        return empty, empty_labels

    arr = np.asarray(windows, dtype=np.float64)
    flat = _fill_nan_2d(arr.reshape(-1, arr.shape[-1]))
    arr = flat.reshape(arr.shape)
    return arr, np.asarray(labels, dtype=np.int32)


def _fill_nan_2d(values):
    arr = np.asarray(values, dtype=np.float64)
    if not np.isnan(arr).any():
        return arr
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        valid = np.isfinite(col)
        if not valid.any():
            col[:] = 0.0
            continue
        first = int(np.argmax(valid))
        col[:first] = col[first]
        valid = np.isfinite(col)
        idx = np.where(valid, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr[:, col_idx] = col[idx]
    return arr


def get_swat_data(base_dir='dataset/SWaT', window_size=100, train_subsample_step=10, test_subsample_step=500):
    """Load sampled SWaT windows and return PyTorch Datasets."""
    normal_path = os.path.join(base_dir, 'normal.csv')
    merged_path = os.path.join(base_dir, 'merged.csv')
    if not os.path.exists(normal_path) or not os.path.exists(merged_path):
        raise FileNotFoundError('SWaT datasets not found in the path.')

    train_header = _read_header(normal_path)
    test_header = _read_header(merged_path)
    feature_indices = _feature_columns(train_header)
    feature_cols = [train_header[idx] for idx in feature_indices]
    test_feature_indices = [test_header.index(col) for col in feature_cols]
    label_idx = test_header.index('Normal/Attack') if 'Normal/Attack' in test_header else len(test_header) - 1

    print(f'Loading sampled SWaT normal.csv windows (stride={train_subsample_step})...')
    X_train_raw, _ = _load_sampled_windows(
        normal_path,
        feature_indices,
        window_size=window_size,
        window_stride=train_subsample_step,
    )
    print(f'Loading sampled SWaT merged.csv windows (stride={test_subsample_step})...')
    X_test_raw, y_true = _load_sampled_windows(
        merged_path,
        test_feature_indices,
        window_size=window_size,
        window_stride=test_subsample_step,
        label_col_idx=label_idx,
    )

    bit_cols = []
    bit_col_indices = []
    train_flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    for idx, col in enumerate(feature_cols):
        unique_cnt = len(np.unique(train_flat[:, idx]))
        if unique_cnt <= 5:
            bit_cols.append(col)
            bit_col_indices.append(idx)
    print(f'Detected {len(bit_cols)} discrete control bit columns out of {len(feature_cols)}: {bit_cols}')

    x_min = train_flat.min(axis=0)
    x_max = train_flat.max(axis=0)
    x_diff = x_max - x_min
    x_diff[x_diff < 1e-6] = 1.0
    X_train_norm = np.clip((X_train_raw - x_min) / x_diff, 0.0, 1.0)
    X_test_norm = np.clip((X_test_raw - x_min) / x_diff, 0.0, 1.0)

    train_ds = SWaTDataset(X_train_norm, bit_col_indices, window_size=window_size, step=1)
    test_ds = SWaTDataset(X_test_norm, bit_col_indices, window_size=window_size, step=1, window_labels=y_true)
    return train_ds, test_ds, y_true, len(feature_cols), len(bit_col_indices)
