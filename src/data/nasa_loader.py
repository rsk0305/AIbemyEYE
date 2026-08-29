# NASA SMAP/MSL telemetry 파일과 anomaly metadata를 Dataset으로 변환하는 로더
import ast
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NASADataset(Dataset):
    def __init__(self, telemetry, bits, window_size=100, step=1):
        self.telemetry = telemetry
        self.bits = bits
        self.window_size = window_size
        self.step = step
        self.length = max(0, (len(telemetry) - window_size) // step + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        x_tele = self.telemetry[start:end]
        x_bits = self.bits[end - 1]
        return {
            'high': torch.tensor(x_tele, dtype=torch.float32).unsqueeze(-1),
            'mid': torch.tensor(x_tele, dtype=torch.float32).unsqueeze(-1),
            'low': torch.tensor([x_tele.mean()], dtype=torch.float32),
            'bits': torch.tensor(x_bits, dtype=torch.float32),
        }


def load_nasa_metadata(csv_path="dataset/NASA/labeled_anomalies.csv"):
    """Load metadata (labels, spacecraft, dimensions) of NASA channels."""
    metadata = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            chan_id = row['chan_id']
            try:
                anom_seqs = ast.literal_eval(row['anomaly_sequences'])
            except Exception:
                anom_seqs = []
            metadata[chan_id] = {
                'spacecraft': row['spacecraft'],
                'anomaly_sequences': anom_seqs,
                'class': row['class'],
                'num_values': int(row['num_values']),
            }
    return metadata


def normalize_nasa_telemetry(
    train_telemetry,
    test_telemetry,
    reference_fraction=1.0,
):
    """Scale telemetry from a declared prefix of the normal train sequence."""
    train = np.asarray(train_telemetry, dtype=np.float64).reshape(-1)
    test = np.asarray(test_telemetry, dtype=np.float64).reshape(-1)
    if train.size == 0 or test.size == 0:
        raise ValueError("train and test telemetry must be non-empty")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("train and test telemetry must contain only finite values")
    if not 0.0 < reference_fraction <= 1.0:
        raise ValueError("reference_fraction must be in (0, 1]")

    reference_rows = max(1, int(train.size * reference_fraction))
    reference = train[:reference_rows]
    minimum = float(np.min(reference))
    maximum = float(np.max(reference))
    scale = maximum - minimum
    constant_reference = scale < 1e-12
    if constant_reference:
        scale = 1.0
        train_scaled = 0.5 + (train - minimum) / scale
        test_scaled = 0.5 + (test - minimum) / scale
    else:
        train_scaled = (train - minimum) / scale
        test_scaled = (test - minimum) / scale
    summary = {
        "method": "fit_reference_minmax",
        "reference_fraction": float(reference_fraction),
        "reference_rows": reference_rows,
        "minimum": minimum,
        "maximum": maximum,
        "constant_reference": bool(constant_reference),
        "constant_reference_center": 0.5 if constant_reference else None,
    }
    return (
        np.clip(train_scaled, 0.0, 1.0),
        np.clip(test_scaled, 0.0, 1.0),
        summary,
    )


def get_nasa_channel_data(
    chan_id,
    base_dir="dataset/NASA/data/data",
    window_size=100,
    normalization_reference_fraction=1.0,
):
    """
    Load train and test telemetry and bits for a single NASA channel,
    apply min-max normalization, and build Datasets.
    """
    train_path = os.path.join(base_dir, "train", f"{chan_id}.npy")
    test_path = os.path.join(base_dir, "test", f"{chan_id}.npy")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files for channel {chan_id} not found.")

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    train_tele = train_data[:, 0]
    test_tele = test_data[:, 0]
    train_bits = train_data[:, 1:]
    test_bits = test_data[:, 1:]

    train_tele_norm, test_tele_norm, normalization_summary = normalize_nasa_telemetry(
        train_tele,
        test_tele,
        reference_fraction=normalization_reference_fraction,
    )

    metadata = load_nasa_metadata()
    anom_seqs = metadata.get(chan_id, {}).get('anomaly_sequences', [])
    y_true = np.zeros(len(test_tele), dtype=np.int32)
    for start_idx, end_idx in anom_seqs:
        y_true[start_idx:end_idx] = 1

    train_ds = NASADataset(train_tele_norm, train_bits, window_size=window_size)
    test_ds = NASADataset(test_tele_norm, test_bits, window_size=window_size)
    train_ds.normalization_summary = normalization_summary
    test_ds.normalization_summary = normalization_summary
    return train_ds, test_ds, y_true, train_bits.shape[1]
