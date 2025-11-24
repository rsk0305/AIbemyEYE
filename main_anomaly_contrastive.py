"""
Skeleton implementation for semi-supervised / contrastive anomaly detection on multi-modal, multi-rate sensor data.
- Python + PyTorch
- Modular: per-modality encoders, temporal fusion, spatial fusion (simple GNN), projection head, decoder, CPC predictor, MIL head
- Losses: InfoNCE (contrastive), CPC predictive loss, reconstruction loss, MIL bag loss
- Scoring: Mahalanobis, reconstruction, CPC error, energy placeholder

This is a skeleton: fill dataset specifics, augmentations, dataloader, and hyperparameters for your environment.
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Utilities ---------------------------------

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# Simple infoNCE implementation
def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """z_a, z_b: (B, D) normalized or not. Compute NT-Xent over batch pairs (positives are a_i<->b_i).
    Returns scalar loss.
    """
    z_a = l2_normalize(z_a, dim=1)
    z_b = l2_normalize(z_b, dim=1)
    batch_size = z_a.shape[0]

    representations = torch.cat([z_a, z_b], dim=0)  # 2B, D
    sim_matrix = torch.matmul(representations, representations.T)  # 2B x 2B
    sim_matrix = sim_matrix / temperature

    # Mask to remove self-similarity
    mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=sim_matrix.device)).float()

    # Positive indices: for i in [0,B): pos = i + B; for i in [B,2B): pos = i - B
    positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)], dim=0).to(sim_matrix.device)

    logits = sim_matrix * mask
    exp_logits = torch.exp(logits) * mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

    # pick positive log-probabilities
    pos_log_prob = log_prob[torch.arange(2 * batch_size), positives]
    loss = -pos_log_prob.mean()
    return loss

# Simple Mahalanobis scorer utility
class MahalanobisScorer:
    def __init__(self, mu: Optional[torch.Tensor] = None, cov_inv: Optional[torch.Tensor] = None):
        self.mu = mu
        self.cov_inv = cov_inv

    def fit(self, embeddings: torch.Tensor):
        # embeddings: (N, D)
        mu = embeddings.mean(dim=0)
        cov = torch.from_numpy(np.cov(embeddings.cpu().numpy(), rowvar=False)).float().to(embeddings.device)
        cov_inv = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
        self.mu = mu
        self.cov_inv = cov_inv

    def score(self, z: torch.Tensor) -> torch.Tensor:
        # returns Mahalanobis distance per row
        assert self.mu is not None and self.cov_inv is not None
        d = z - self.mu.unsqueeze(0)
        left = torch.matmul(d, self.cov_inv)
        m = (left * d).sum(dim=1)
        return m

# ----------------------------- Data --------------------------------------
class MultiModalWindowDataset(Dataset):
    """Placeholder dataset. Implement __getitem__ to return a time-window bag of modalities.
    Return a dict with keys per modality, e.g.:
      {
        'high': tensor (T_high, C_high),
        'mid': tensor (T_mid, C_mid),
        'low': tensor (T_low, C_low),
        'bits': tensor (B,) or (T_low, B),
        'bag_label': 0 or 1 (optional weak label at window level)
      }
    """
    def __init__(self, windows: list, transform=None):
        self.windows = windows
        self.transform = transform

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        sample = self.windows[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# ------------------------ Modality Encoders -------------------------------
class HighRateEncoder(nn.Module):
    """1D-CNN/TCN style encoder for high-rate sensors (e.g., 2kHz)
    Input: (B, T_high, C)
    Output: (B, T_out, D)
    """
    def __init__(self, in_ch: int, hidden: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # pool across time to produce per-window vector
        )
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        h = self.net(x).squeeze(-1)
        return self.fc(h)  # (B, out_dim)

class MidRateEncoder(nn.Module):
    """Temporal conv for medium rate (200Hz) that preserves some time resolution"""
    def __init__(self, in_ch: int, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        h = F.relu(self.conv(x))
        h = self.pool(h).squeeze(-1)
        return h

class LowRateEncoder(nn.Module):
    """For low-rate sensors or event-based inputs"""
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, features) or (B, T, features) - handle both
        if x.dim() == 3:
            # average pool time
            x = x.mean(dim=1)
        return self.mlp(x)

class BitFieldEncoder(nn.Module):
    """Encode complex bitfield/binary channels via embedding or small MLP"""
    def __init__(self, bit_len: int, out_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(bit_len, 128), nn.ReLU(), nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, bit_len) or (B, T, bit_len)
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.fc(x)

# ------------------------ Spatial Fusion (simple GNN) ----------------------
class SimpleGNN(nn.Module):
    """Simple graph convolution using adjacency matrix multiply.
    Node features shape: (B, N_nodes, D)
    Adj shape: (N_nodes, N_nodes) or (B, N, N)
    """
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, D)
        h = self.lin(x)
        if adj is None:
            # assume fully connected
            agg = h.mean(dim=1, keepdim=True).expand_as(h)
        else:
            # adj: (N, N) or (B, N, N)
            if adj.dim() == 2:
                agg = torch.matmul(adj, h)  # (N,N) * (B,N,H) broadcasting error; handle per batch
                # bring to batch dimension
                agg = agg.unsqueeze(0).expand(h.shape[0], -1, -1)
            else:
                agg = torch.matmul(adj, h)
        h2 = F.relu(agg)
        out = self.lin2(h2)
        return out

# ------------------------ Fusion Module ----------------------------------
class FusionTransformer(nn.Module):
    def __init__(self, input_dim: int, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Seq, D) -> Transformer expects (Seq, B, D)
        x_t = x.permute(1, 0, 2)
        out = self.transformer(x_t)
        out = out.permute(1, 0, 2)
        # optionally pool
        return out.mean(dim=1)

# ------------------------ Projection / Heads ------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    """Optional decoder for reconstruction (simple MLP)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class CPCPredictor(nn.Module):
    """Predict future embeddings from current embedding"""
    def __init__(self, in_dim: int, pred_steps: int = 3):
        super().__init__()
        self.pred_steps = pred_steps
        self.net = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(pred_steps)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D) -> returns (B, pred_steps, D)
        preds = [m(z).unsqueeze(1) for m in self.net]
        return torch.cat(preds, dim=1)

class MILHead(nn.Module):
    """Bag-level attention pooling for MIL weak labels"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, inst_feats: torch.Tensor) -> torch.Tensor:
        # inst_feats: (B, T, D)
        scores = self.attn(inst_feats)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # temporal attention
        bag_repr = (weights * inst_feats).sum(dim=1)  # (B, D)
        bag_score = torch.sigmoid(bag_repr.mean(dim=1))  # (B,) simple
        return bag_score, bag_repr

# ------------------------ Full Model Wrapper ------------------------------
class MultiModalAnomalyModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        # instantiate modality encoders
        self.high_enc = HighRateEncoder(in_ch=config['high_ch'], out_dim=config['enc_dim'])
        self.mid_enc = MidRateEncoder(in_ch=config['mid_ch'], out_dim=config['enc_dim'])
        self.low_enc = LowRateEncoder(in_dim=config['low_dim'], out_dim=config['enc_dim']//2)
        self.bit_enc = BitFieldEncoder(bit_len=config['bit_len'], out_dim=config['enc_dim']//4)

        # optional graph fusion
        self.gnn = SimpleGNN(in_dim=config['enc_dim'], hidden=128, out_dim=config['enc_dim'])

        # fusion transformer
        # assume we'll concat modality vectors into sequence of length M
        self.fusion = FusionTransformer(input_dim=config['enc_dim'])

        # heads
        self.proj = ProjectionHead(in_dim=config['enc_dim'], proj_dim=config['proj_dim'])
        self.decoder = Decoder(in_dim=config['enc_dim'], out_dim=config['recon_dim'])
        self.cpc = CPCPredictor(in_dim=config['enc_dim'], pred_steps=config.get('pred_steps', 3))
        self.mil = MILHead(in_dim=config['enc_dim'])
        # classifier for energy / fine-tune
        self.classifier = nn.Linear(config['enc_dim'], 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # batch expected keys: 'high', 'mid', 'low', 'bits', optionally 'adj'
        z_high = self.high_enc(batch['high'])  # (B, D)
        z_mid = self.mid_enc(batch['mid'])
        z_low = self.low_enc(batch['low'])
        z_bits = self.bit_enc(batch['bits'])

        # stack into (B, M, D)
        # first unify dim by linear projections if needed
        features = torch.stack([z_high, z_mid, z_low, z_bits], dim=1)  # (B, 4, D)

        # spatial fusion via GNN (if sensors as nodes - here simplified)
        gnn_out = self.gnn(features)  # (B, 4, D)

        fused = self.fusion(gnn_out)  # (B, D)

        proj = self.proj(fused)  # (B, proj_dim)
        proj_norm = l2_normalize(proj, dim=1)

        recon = self.decoder(fused)
        preds = self.cpc(fused)

        bag_score, bag_repr = self.mil(features)  # (B,), (B,D)

        logits = self.classifier(fused).squeeze(-1)

        return {
            'fused': fused,
            'proj': proj_norm,
            'recon': recon,
            'cpc_preds': preds,
            'bag_score': bag_score,
            'logits': logits,
            'inst_feats': features,
        }

# ------------------------ Losses & Training Skeleton ----------------------
import numpy as np


def cpc_loss_fn(preds: torch.Tensor, true_future: torch.Tensor) -> torch.Tensor:
    # preds: (B, S, D) , true_future: (B, S, D)
    return F.mse_loss(preds, true_future)


def mil_bce_loss(bag_score: torch.Tensor, bag_label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(bag_score, bag_label.float())


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):
        self.model = model.to(device)
        self.device = device
        self.optim = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-3), weight_decay=1e-4)
        self.config = config

    def pretrain_contrastive(self, dataloader: DataLoader, epochs: int = 100):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                # create two augmented views per modality
                batch_a = self._augment_batch(batch)
                batch_b = self._augment_batch(batch)
                batch_a = {k: v.to(self.device) for k, v in batch_a.items()}
                batch_b = {k: v.to(self.device) for k, v in batch_b.items()}

                out_a = self.model(batch_a)
                out_b = self.model(batch_b)
                z_a = out_a['proj']
                z_b = out_b['proj']

                loss_con = info_nce_loss(z_a, z_b, temperature=self.config.get('tau', 0.07))

                # optional CPC loss using true future embeddings if available in batch
                loss = loss_con
                if 'future' in batch:
                    # forward true future through model (simple approach)
                    future = batch['future'].to(self.device)
                    # this is placeholder: compute target future fused embedding via a separate pass
                    with torch.no_grad():
                        future_out = self.model({'high': future['high'], 'mid': future['mid'], 'low': future['low'], 'bits': future['bits']})
                        true_fused = future_out['fused']
                    preds = out_a['cpc_preds']
                    loss += self.config.get('lambda_cpc', 1.0) * cpc_loss_fn(preds, true_fused.unsqueeze(1).repeat(1, preds.shape[1], 1))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def finetune_mil(self, dataloader: DataLoader, epochs: int = 50):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                loss_mil = mil_bce_loss(outputs['bag_score'], batch['bag_label'])
                # optional supervised BCE for small labelled instances
                if 'instance_label' in batch:
                    inst_lbl = batch['instance_label'].to(self.device)
                    logits = outputs['logits']
                    loss_sup = F.binary_cross_entropy_with_logits(logits, inst_lbl.float())
                    loss = loss_mil + self.config.get('lambda_sup', 1.0) * loss_sup
                else:
                    loss = loss_mil
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def _augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Implement domain-specific augmentations: jitter, scaling, masking
        out = {}
        for k, v in batch.items():
            if k in ['bag_label', 'instance_label', 'future']:
                out[k] = v
                continue
            x = v.clone()
            # small gaussian noise
            if x.is_floating_point():
                x = x + 0.005 * torch.randn_like(x)
            out[k] = x
        return out

# ------------------------ Scoring / Thresholding --------------------------

def compute_scores(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    scores = []
    recons = []
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'bag_label'}
            out = model(batch)
            emb = out['fused']
            recon = out['recon']
            embeddings.append(emb.cpu())
            recons.append(((batch['high'].reshape(batch['high'].shape[0], -1) - recon).pow(2).mean(dim=1)).cpu())
    embeddings = torch.cat(embeddings, dim=0)
    recons = torch.cat(recons, dim=0)

    # fit Mahalanobis
    mu = embeddings.mean(dim=0)
    cov = torch.from_numpy(np.cov(embeddings.numpy(), rowvar=False)).float()
    cov_inv = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0]))
    d = embeddings - mu.unsqueeze(0)
    left = d.matmul(cov_inv)
    maha = (left * d).sum(dim=1)

    # combine scores
    s = (maha - maha.mean()) / (maha.std() + 1e-9) + (recons - recons.mean()) / (recons.std() + 1e-9)
    return s.numpy()

# Placeholder EVT thresholding (recommend using scipy / pyculiarity etc.)
def evt_threshold(scores: np.ndarray, q: float = 0.995) -> float:
    # simple percentile fallback
    return float(np.quantile(scores, q))

# ------------------------ Example usage ----------------------------------
if __name__ == '__main__':
    # Fill in with actual data
    config = {
        'high_ch': 6,
        'mid_ch': 4,
        'low_dim': 8,
        'bit_len': 16,
        'enc_dim': 256,
        'proj_dim': 128,
        'recon_dim': 6 * 100,  # example flatten high-rate to reconstruct
        'pred_steps': 3,
        'tau': 0.07,
        'lambda_cpc': 1.0,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalAnomalyModel(config)
    trainer = Trainer(model, device, config)

    # Create DataLoader with your MultiModalWindowDataset
    dummy_windows = []  # populate this with real windows
    ds = MultiModalWindowDataset(dummy_windows)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=lambda x: x)

    # trainer.pretrain_contrastive(dl, epochs=10)
    # ... then fine-tune MIL if bag labels available

    print('Skeleton ready. Fill dataset and hyperparams and run training.')
