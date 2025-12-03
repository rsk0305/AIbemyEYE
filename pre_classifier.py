"""
Pre-classifier for sensor data type classification.
Classifies sensor signals into: 1word, 2word, or bits.
Uses deep learning model with rate-specific encoders and graph neural network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
from typing import List, Dict, Any, Tuple
import os

# ========================
# Feature Extraction Utilities
# ========================

def extract_features_numpy(signal: np.ndarray, rate_hz: int = 200) -> Dict[str, float]:
    """
    Extract statistical and signal-based features from a 1D time series.
    
    Args:
        signal: 1D numpy array
        rate_hz: sampling rate in Hz
    
    Returns:
        dict of features for classification
    """
    sig = np.asarray(signal, dtype=np.float64)
    if len(sig) == 0:
        return {}
    
    features = {}
    
    # Basic statistics
    features['mean'] = float(np.mean(sig))
    features['std'] = float(np.std(sig))
    features['min'] = float(np.min(sig))
    features['max'] = float(np.max(sig))
    features['range'] = float(np.max(sig) - np.min(sig))
    features['median'] = float(np.median(sig))
    
    # Amplitude-based
    features['skewness'] = float(np.mean((sig - features['mean'])**3) / (features['std']**3 + 1e-8))
    features['kurtosis'] = float(np.mean((sig - features['mean'])**4) / (features['std']**4 + 1e-8))
    
    # Smoothness: autocorrelation at lag 1
    if len(sig) > 1:
        c0 = np.sum((sig - features['mean'])**2)
        c1 = np.sum((sig[:-1] - features['mean']) * (sig[1:] - features['mean']))
        features['autocorr_lag1'] = float(c1 / (c0 + 1e-8))
    else:
        features['autocorr_lag1'] = 0.0
    
    # Uniqueness: ratio of unique values
    unique_vals = len(np.unique(sig))
    features['unique_ratio'] = float(unique_vals / (len(sig) + 1e-8))
    
    # Entropy: measure of randomness
    hist, _ = np.histogram(sig, bins=min(32, len(sig)), range=(sig.min(), sig.max()))
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    features['entropy'] = float(-np.sum(probs * np.log2(probs + 1e-8)))
    
    # Difference statistics
    if len(sig) > 1:
        diff = np.diff(sig)
        features['diff_mean'] = float(np.mean(diff))
        features['diff_std'] = float(np.std(diff))
        features['diff_max'] = float(np.max(np.abs(diff)))
    else:
        features['diff_mean'] = 0.0
        features['diff_std'] = 0.0
        features['diff_max'] = 0.0
    
    # Bit-field specific: check if signal contains only 0s/1s or few distinct values
    unique_count = len(np.unique(sig))
    features['unique_count'] = float(unique_count)
    
    # Check for periodic bit pattern
    if len(sig) > 16:
        # Look for repeating patterns in bit positions
        bit_transitions = float(np.sum(np.abs(np.diff(sig)) > 0))
        features['bit_transitions'] = bit_transitions
    else:
        features['bit_transitions'] = 0.0
    
    return features


def classify_heuristic(features: Dict[str, float]) -> Tuple[str, float]:
    """
    Heuristic-based classification using feature statistics.
    
    Returns:
        (predicted_type, confidence)
        predicted_type: '1word', '2word', or 'bits'
        confidence: float between 0 and 1
    """
    unique_ratio = features.get('unique_ratio', 0.0)
    entropy = features.get('entropy', 0.0)
    autocorr = features.get('autocorr_lag1', 0.0)
    range_val = features.get('range', 0.0)
    std = features.get('std', 0.0)
    unique_count = features.get('unique_count', 0.0)
    
    # Rule 1: Bits have very low unique ratio and entropy
    # (because they're mostly 0s with a few bit positions toggling)
    if unique_ratio < 0.05 and entropy < 2.0 and unique_count < 20:
        return 'bits', 0.95
    
    # Rule 2: 2word has LARGE range and std (32-bit counter with drift)
    # Typically higher range than 1word
    if range_val > 30000 and std > 5000 and autocorr > 0.7:
        return '2word', 0.90
    
    # Rule 3: 1word has moderate range, moderate-to-high autocorr (smooth signal)
    # but smaller range than 2word
    if range_val <= 30000 and autocorr > 0.6 and entropy > 2.0:
        return '1word', 0.85
    
    # Default fallback
    if autocorr > 0.5:
        return '1word', 0.60
    else:
        return 'bits', 0.50


# ========================
# Deep Learning Components
# ========================

class RateEncoderForClassification(nn.Module):
    """Simple CNN encoder for feature extraction from time series."""
    
    def __init__(self, emb_dim=64, conv_channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_channels, emb_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) or (T,)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
        
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)  # (B, conv_channels)
        vec = self.fc(h)  # (B, emb_dim)
        return vec


class TypeClassifierNN(nn.Module):
    """Deep learning classifier for sensor type (1word/2word/bits)."""
    
    def __init__(self, emb_dim=64, hidden_dim=128, num_classes=3, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.emb_dim = emb_dim
        
        # Encoder for rate=200
        self.encoder = RateEncoderForClassification(emb_dim=emb_dim, conv_channels=32)
        
        # Classification head
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        
        # Class names: 0='1word', 1='2word', 2='bits'
        self.class_names = ['1word', '2word', 'bits']
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) tensor or single signal
        
        Returns:
            logits: (B, 3) class logits
        """
        emb = self.encoder(x)  # (B, emb_dim)
        h = F.relu(self.fc1(emb))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)
        logits = self.fc_out(h)  # (B, 3)
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[str, float]:
        """
        Predict type for a single signal with confidence.
        
        Args:
            x: 1D tensor or numpy array
        
        Returns:
            (predicted_type, confidence)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if x.dim() == 1:
                x = x.unsqueeze(0)  # (1, T)
            
            logits = self(x)  # (1, 3)
            probs = F.softmax(logits, dim=1)[0]  # (3,)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
        return self.class_names[pred_idx], confidence

class AttentionPooling(nn.Module):
    """Self-attention pooling layer."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # (B, T, D)
        weights = torch.softmax(self.attn(x), dim=1)  # (B,T,1)
        return torch.sum(weights * x, dim=1)  # (B,D)

class ImprovedTypeClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1,
        emb_dim=128,
        lstm_dim=128,
        num_transformer_layers=2,
        num_classes=3,
        proj_dim=64,
        device = 'cpu'
    ):
        super().__init__()
        
        self.device = torch.device(device)
        # --- CNN Feature Extractor ---
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, emb_dim, 5, padding=2),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
        )
        
        # --- BiLSTM for sequential understanding ---
        self.lstm = nn.LSTM(
            emb_dim, lstm_dim, bidirectional=True, batch_first=True
        )
        
        # --- Attention Pooling ---
        self.pool = AttentionPooling(lstm_dim * 2)
        
        # --- Transformer Encoder for structural invariance ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_dim * 2,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        
        # --- Classification Head ---
        self.class_head = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim),
            nn.GELU(),
            nn.LayerNorm(lstm_dim),
            nn.Dropout(0.2),
            nn.Linear(lstm_dim, num_classes)
        )

        # --- Contrastive Projection Head (optional) ---
        self.projector = nn.Sequential(
            nn.Linear(lstm_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x, return_projection=False):
        """
        x: (B,T)
        return_projection: If True returns embedding for contrastive learning.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,T)
        
        # CNN → (B, emb_dim, T)
        h = self.conv(x)
        
        # reorder for LSTM → (B,T,emb_dim)
        h = h.transpose(1, 2)
        
        # BiLSTM
        h, _ = self.lstm(h)

        # Transformer Encoder for token-level refinement
        h = self.transformer(h)

        # Self-attention pooling → (B,D)
        pooled = self.pool(h)

        logits = self.class_head(pooled)

        if return_projection:
            proj = self.projector(pooled)
            return logits, proj

        return logits

# ========================
# Pre-classifier Integration
# ========================


class SensorPreClassifier:
    """
    Main interface for sensor type classification.
    Combines heuristic and deep learning approaches.
    """
    
    def __init__(self, use_deep_learning=True, device='cpu'):
        self.use_deep_learning = use_deep_learning
        self.device = device
        self.model = None
        
        if use_deep_learning:
            self.model = TypeClassifierNN(emb_dim=64, hidden_dim=128, num_classes=3, device=device)
            self.model.to(device)
    
    def classify_signal(self, signal: np.ndarray, rate_hz: int = 200, 
                       use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Classify a sensor signal into type: '1word', '2word', or 'bits'.
        
        Args:
            signal: 1D numpy array of sensor values
            rate_hz: sampling rate (default 200 Hz)
            use_hybrid: if True, combine heuristic and DL predictions
        
        Returns:
            dict with keys:
                - 'type': predicted type ('1word', '2word', 'bits')
                - 'confidence': float [0, 1]
                - 'heuristic_type': heuristic result
                - 'heuristic_conf': heuristic confidence
                - 'dl_type': DL result (if available)
                - 'dl_conf': DL confidence (if available)
                - 'features': extracted features dict
        """
        result = {}
        
        # Extract features
        features = extract_features_numpy(signal, rate_hz)
        result['features'] = features
        
        # Heuristic classification
        h_type, h_conf = classify_heuristic(features)
        result['heuristic_type'] = h_type
        result['heuristic_conf'] = h_conf
        
        # Deep learning classification
        if self.use_deep_learning and self.model is not None:
            sig_tensor = torch.from_numpy(np.asarray(signal, dtype=np.float32)).to(self.device)
            dl_type, dl_conf = self.model.predict(sig_tensor)
            result['dl_type'] = dl_type
            result['dl_conf'] = dl_conf
            
            # Hybrid: weighted average
            if use_hybrid:
                # Weight heuristic more on clear cases, DL more on ambiguous
                heuristic_weight = h_conf
                dl_weight = dl_conf
                total_weight = heuristic_weight + dl_weight
                
                # Favor agreement; if both agree, boost confidence
                if h_type == dl_type:
                    final_conf = min(1.0, (heuristic_weight + dl_weight) / 2.0 + 0.1)
                    result['type'] = h_type
                else:
                    # Weighted vote
                    final_conf = max(heuristic_weight, dl_weight)
                    result['type'] = h_type if heuristic_weight > dl_weight else dl_type
                
                result['confidence'] = final_conf
                result['method'] = 'hybrid'
            else:
                result['type'] = dl_type
                result['confidence'] = dl_conf
                result['method'] = 'deep_learning'
        else:
            result['type'] = h_type
            result['confidence'] = h_conf
            result['method'] = 'heuristic'
        
        return result
    
    def train_on_scenes(self, scenes: List[List[Dict]], device='cpu', 
                       epochs=10, lr=1e-3, batch_size=32):
        """
        Train the deep learning model on synthetic scenes.
        
        Args:
            scenes: list of scenes, each scene is list of sensor dicts
            device: 'cpu' or 'cuda'
            epochs: number of training epochs
            lr: learning rate
            batch_size: batch size for training
        """
        if not self.use_deep_learning:
            print("[Warning] Deep learning disabled; training skipped.")
            return
        
        # Collect (signal, type_label) pairs
        samples = []
        for scene in scenes:
            for sensor in scene:
                signal = sensor['raw']
                sensor_type = sensor['type']
                
                # Convert type string to label
                if sensor_type == '1word':
                    label = 0
                elif '2word' in sensor_type:
                    label = 1
                elif sensor_type == 'bits':
                    label = 2
                else:
                    continue
                
                # Convert to numpy if needed
                if isinstance(signal, torch.Tensor):
                    signal = signal.cpu().numpy()
                
                samples.append((signal, label))
        
        print(f"[Pre-classifier] Training on {len(samples)} samples")
        
        # Create dataset
        class SignalDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sig, label = self.samples[idx]
                sig_t = torch.from_numpy(np.asarray(sig, dtype=np.float32))
                return sig_t, label
        
        dataset = SignalDataset(samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for ep in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_sig, batch_label in loader:
                batch_sig = batch_sig.to(device)
                batch_label = batch_label.to(device)
                
                logits = self.model(batch_sig)
                loss = criterion(logits, batch_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                correct += (pred == batch_label).sum().item()
                total += batch_label.size(0)
            
            acc = correct / total if total > 0 else 0.0
            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
            print(f"[Pre-classifier] Epoch {ep+1}/{epochs} loss={avg_loss:.4f} acc={acc:.4f}")
    
    def validate_on_scenes(self, scenes: List[List[Dict]], device='cpu') -> Dict[str, float]:
        """
        Validate classifier on scenes and return metrics.
        
        Returns:
            dict with keys: 'accuracy', 'precision', 'recall', 'f1', etc.
        """
        # Collect predictions and ground truth
        predictions = []
        ground_truth = []
        
        for scene in scenes:
            for sensor in scene:
                signal = sensor['raw']
                sensor_type = sensor['type']
                
                # Convert type to label
                if sensor_type == '1word':
                    label = 0
                elif '2word' in sensor_type:
                    label = 1
                elif sensor_type == 'bits':
                    label = 2
                else:
                    continue
                
                # Convert signal to numpy
                if isinstance(signal, torch.Tensor):
                    signal = signal.cpu().numpy()
                
                # Classify
                result = self.classify_signal(signal)
                pred_type = result['type']
                
                # Convert prediction to label
                if pred_type == '1word':
                    pred_label = 0
                elif pred_type == '2word':
                    pred_label = 1
                else:  # 'bits'
                    pred_label = 2
                
                predictions.append(pred_label)
                ground_truth.append(label)
        
        if len(predictions) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Accuracy
        accuracy = np.mean(predictions == ground_truth)
        
        # Per-class metrics
        metrics = {'accuracy': float(accuracy)}
        
        for label, class_name in enumerate(['1word', '2word', 'bits']):
            mask = ground_truth == label
            if mask.sum() == 0:
                continue
            
            tp = ((predictions == label) & mask).sum()
            fp = ((predictions == label) & ~mask).sum()
            fn = ((predictions != label) & mask).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'{class_name}_precision'] = float(precision)
            metrics[f'{class_name}_recall'] = float(recall)
            metrics[f'{class_name}_f1'] = float(f1)
        
        return metrics


# ========================
# Demo / Main
# ========================

if __name__ == "__main__":
    import sys
    
    # Import data generator
    sys.path.insert(0, '/workspaces/AIbemyEYE')
    from main_data_generator import generate_multimodal_data_advanced
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Generate synthetic scenes
    print("=== Generating synthetic data ===")
    scenes_train = []
    for seed in range(50):
        sensors_by_rate, sensors_flat = generate_multimodal_data_advanced(
            num_sensors=10,
            duration_sec=1.0,
            rates=[200],  # Single rate: 200 Hz
            prob_1word=0.5,
            prob_2word=0.25,
            prob_bits=0.25,
            seed=seed
        )
        # Convert to scene format compatible with pre_classifier
        scene = []
        for s in sensors_flat:
            sensor_dict = {
                'id': s['id'],
                'type': s['meta']['type'],
                'raw_rate': s['raw_rate'],
                'raw': s['raw_signal'].astype(np.float32),
                'meta': s['meta']
            }
            scene.append(sensor_dict)
        scenes_train.append(scene)
    
    scenes_val = []
    for seed in range(50, 60):
        sensors_by_rate, sensors_flat = generate_multimodal_data_advanced(
            num_sensors=10,
            duration_sec=1.0,
            rates=[200],
            prob_1word=0.5,
            prob_2word=0.25,
            prob_bits=0.25,
            seed=seed
        )
        scene = []
        for s in sensors_flat:
            sensor_dict = {
                'id': s['id'],
                'type': s['meta']['type'],
                'raw_rate': s['raw_rate'],
                'raw': s['raw_signal'].astype(np.float32),
                'meta': s['meta']
            }
            scene.append(sensor_dict)
        scenes_val.append(scene)
    
    # Initialize classifier
    print("\n=== Initializing Pre-Classifier ===")
    classifier = SensorPreClassifier(use_deep_learning=True, device=device)
    
    # Train on scenes
    print("\n=== Training classifier on scenes ===")
    classifier.train_on_scenes(scenes_train, device=device, epochs=15, lr=1e-3, batch_size=64)
    
    # Validate
    print("\n=== Validating on validation set ===")
    val_metrics = classifier.validate_on_scenes(scenes_val, device=device)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Demo on single samples
    print("\n=== Demo: Classify single samples ===")
    for scene_idx, scene in enumerate(scenes_val[:3]):
        print(f"\nScene {scene_idx}:")
        for sensor_idx, sensor in enumerate(scene[:3]):
            signal = sensor['raw']
            true_type = sensor['type']
            
            result = classifier.classify_signal(signal)
            pred_type = result['type']
            confidence = result['confidence']
            
            status = "✓" if pred_type == true_type else "✗"
            print(f"  {status} Sensor {sensor_idx}: True={true_type:12} Pred={pred_type:12} Conf={confidence:.3f}")
            print(f"      Method: {result['method']}")
            if 'heuristic_type' in result:
                print(f"      Heuristic: {result['heuristic_type']} ({result['heuristic_conf']:.3f})")
            if 'dl_type' in result:
                print(f"      DL: {result['dl_type']} ({result['dl_conf']:.3f})")
    
    print("\n=== Pre-classifier demo complete ===")
