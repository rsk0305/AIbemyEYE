"""
type_graph_model.py
- Full runnable example:
  - SceneDataset (synthetic)
  - TypeGraphModel (sensor encoder + transformer attention across sensors)
  - NT-Xent contrastive pretrain (per-scene, batch_size=1)
  - Hybrid finetune (contrastive + supervised node CE + edge BCE)
  - Simple evaluation metrics
"""

import math, random
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Synthetic Scene Generator (for demo/training)
# Each scene is a list of sensors, each sensor: dict {id, type, raw (1D numpy), meta}
# types: '1word', '2word_lsb', '2word_msb', 'bits'
# We'll produce variable N per scene.
# -------------------------
def synthetic_scene(num_sensors_min=6, num_sensors_max=12, duration_samples=512, p1=0.6, p2=0.2, p_bits=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    sensors = []
    sid = 0
    i = 0
    while i < np.random.randint(num_sensors_min, num_sensors_max+1):
        r = np.random.rand()
        if r < p1:
            # 1word: sinusoid + noise -> CPU tensor로 저장 (device 이동 안함)
            t = np.arange(duration_samples)
            f = np.random.uniform(1, 8)
            raw = 1000*np.sin(2*np.pi*f*t/duration_samples + np.random.rand()*2*np.pi)
            raw += np.random.normal(0, 20, size=duration_samples)
            # torch tensor로 생성 (CPU에만 저장, device 이동 X)
            raw_tensor = torch.from_numpy(raw.astype(np.float32))
            sensors.append({'id': sid, 'type':'1word', 'raw': raw_tensor, 'meta':{}})
            sid += 1; i += 1
        elif r < p1 + p2:
            # 2word pair
            base = np.random.randint(0, 1<<20)
            noise = np.cumsum(np.random.normal(0, 50, duration_samples))
            raw32 = (base + noise).astype(np.int64) & 0xFFFFFFFF
            lsb = torch.from_numpy(((raw32 & 0xFFFF).astype(np.float32)))
            msb = torch.from_numpy((((raw32 >> 16) & 0xFFFF).astype(np.float32)))
            sensors.append({'id': sid, 'type':'2word_lsb', 'raw': lsb, 'meta':{'pair_id': sid+1, 'raw32': raw32}})
            sid += 1
            sensors.append({'id': sid, 'type':'2word_msb', 'raw': msb, 'meta':{'pair_id': sid-1, 'raw32': raw32}})
            sid += 1
            i += 2
        else:
            # bits
            width = np.random.randint(1,6)
            start = np.random.randint(0, 16-width+1)
            vals = np.cumsum((np.random.rand(duration_samples) < 0.01).astype(int))
            word = ((vals & ((1<<width)-1)) << start).astype(np.float32)
            word_tensor = torch.from_numpy(word)
            sensors.append({'id': sid, 'type':'bits', 'raw': word_tensor, 'meta':{'bit_start':start, 'bit_end':start+width-1}})
            sid += 1; i += 1
    return sensors

# -------------------------
# Dataset wrapper / collate
# -------------------------
class SceneDataset(Dataset):
    def __init__(self, scenes: List[List[Dict[str,Any]]]):
        self.scenes = scenes
    def __len__(self):
        return len(self.scenes)
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        # convert raw to torch tensors (but keep list-of-sensors)
        scene_copy = []
        for s in scene:
            scene_copy.append({
                'id': s['id'],
                'type': s['type'],
                'raw': torch.from_numpy(np.asarray(s['raw'], dtype=np.float32)),
                'meta': s.get('meta', {})
            })
        return scene_copy

def collate_scene(batch):
    # batch is list of scenes (we will use batch_size=1 for simplicity)
    return batch[0]

# -------------------------
# Utilities: NT-Xent (contrastive) adapted to variable N per scene
# We'll create two augmentations per sensor, compute 2N embeddings and use NT-Xent
# -------------------------
def augment_tensor_1d(x: torch.Tensor, jitter_scale=0.01, mask_ratio=0.03):
    
    # safe conversion: accept numpy or tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        
    x = x.clone()
    # jitter relative to std
    std = (x.std().item() if x.numel()>1 else 1.0)
    x = x + torch.randn_like(x) * (jitter_scale * (std+1e-6))
    # mask a small segment
    L = x.shape[0]
    mlen = max(1, int(L * mask_ratio))
    if L > mlen:
        start = random.randint(0, L-mlen)
        x[start:start+mlen] = 0.0
    return x

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature=0.1):
    # z1, z2: (N, E) embeddings for same scene sensors
    device = z1.device
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)  # (2N, E)
    N2 = z.shape[0]
    sim = torch.matmul(z, z.T) / temperature  # (2N,2N)
    # mask = identity excluded
    mask = (~torch.eye(N2, dtype=torch.bool, device=device)).float()
    exp_sim = torch.exp(sim) * mask
    denom = exp_sim.sum(dim=1, keepdim=True)
    # positives: for i in [0..N-1], positive index is i+N, and vice versa
    pos_idx = torch.arange(z1.shape[0], device=device)
    pos = torch.cat([pos_idx + z1.shape[0], pos_idx], dim=0)
    log_prob = sim - torch.log(denom + 1e-12)
    loss = -log_prob[torch.arange(N2, device=device), pos].mean()
    return loss

# -------------------------
# Model: TypeGraphModel
# - Shared temporal encoder per sensor (Conv1D -> pooling)
# - Project to embedding E
# - Cross-sensor Transformer (attends across N sensors) -> updated embeddings
# - Node classifier (N x 3)
# - Edge head (pairwise MLP on [h_i || h_j]) -> (N,N) logits
# - Projection head for contrastive
# -------------------------
class SensorTemporalEncoder(nn.Module):
    def __init__(self, in_ch=1, conv_channels=64, time_pooled=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, conv_channels//2, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels//2),
            nn.GELU(),
            nn.Conv1d(conv_channels//2, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(time_pooled)  # (B, C, time_pooled)
        )
        self.time_pooled = time_pooled
        self.conv_channels = conv_channels

    def forward(self, x):
        # x: (B*N, 1, D) or (batch=1*N sensors)
        return self.net(x)  # (B*N, C, L)

class TypeGraphModel(nn.Module):
    def __init__(self, emb_dim=128, conv_channels=64, time_pooled=16, transformer_layers=2, nhead=4, device='cpu'):
        super().__init__()
        self.device = device
        self.sensor_enc = SensorTemporalEncoder(in_ch=1, conv_channels=conv_channels, time_pooled=time_pooled)
        flat_dim = conv_channels * time_pooled
        self.proj = nn.Linear(flat_dim, emb_dim)
        # transformer over sensor tokens (tokens = sensors)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.trf = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        # node classifier
        self.node_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//1),
            nn.ReLU(),
            nn.Linear(emb_dim, 3)
        )
        # pairwise edge head (applied on concatenated features)
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )
        # projection head for contrastive
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def encode_scene(self, scene: List[Dict[str,Any]]):
        """
        scene: list of sensor dicts (with 'raw' tensor shape (D,))
        returns:
           H: (N, emb_dim) sensor embeddings (pre-transformer)
           H_trf: (N, emb_dim) transformed embeddings
        """
        N = len(scene)
        # on-the-fly 이동: forward 시점에만 device로 이동
        raws = [s['raw'].to(self.device) for s in scene]  # list of tensors (D,)
        # pack into (N,1,D)
        X = torch.stack([r.unsqueeze(0) for r in raws], dim=0)  # (N,1,D)
        # convert to (N,1,D) but sensor_enc expects (B*N,1,D)
        X = X.view(N, 1, X.shape[-1])
        # apply encoder: returns (N, C, L)
        h = self.sensor_enc(X)  # (N, C, L)
        h = h.view(N, -1)  # (N, C*L)
        z = self.proj(h)  # (N, E)
        # transformer expects batch dimension; we treat sensors as sequence length with batch size =1
        z_in = z.unsqueeze(0)  # (1, N, E)
        z_trf = self.trf(z_in).squeeze(0)  # (N, E)
        return z, z_trf

    def forward_scene(self, scene: List[Dict[str,Any]]):
        z, h = self.encode_scene(scene)  # both (N,E)
        # node logits from transformer output (use h)
        node_logits = self.node_head(h)  # (N,3)
        # edge logits: compute pairwise concatenation
        N = h.shape[0]
        hi = h.unsqueeze(1).expand(N, N, -1)  # (N,N,E)
        hj = h.unsqueeze(0).expand(N, N, -1)
        pair = torch.cat([hi, hj], dim=-1).view(N*N, -1)
        edge_logits = self.edge_mlp(pair).view(N, N)  # (N,N)
        # projection embeddings for contrastive
        proj = self.projector(z)  # use pre-trf embedding for contrastive stability
        return {
            'node_logits': node_logits,   # (N,3)
            'edge_logits': edge_logits,   # (N,N)
            'proj': proj                  # (N, E)
        }

# -------------------------
# Ground-truth builder for supervised losses (from synthetic scene)
# - node_types: (N,) long tensor (0/1/2)
# - pair_adj: (N,N) float tensor  (1 if pair)
# -------------------------
def build_gt_from_scene(scene: List[Dict[str,Any]]):
    N = len(scene)
    node_types = []
    pair_adj = np.zeros((N,N), dtype=np.float32)
    for i,s in enumerate(scene):
        t = s['type']
        if t == '1word':
            node_types.append(0)
        elif t.startswith('2word'):
            node_types.append(1)
            # if this node has pair_id in meta
            pair_id = s['meta'].get('pair_id', None)
            if pair_id is not None:
                # find index j with id == pair_id
                for j, ss in enumerate(scene):
                    if ss['id'] == pair_id:
                        pair_adj[i, j] = 1.0
        else:
            node_types.append(2)
    return {
        'node_types': torch.tensor(node_types, dtype=torch.long),
        'pair_adj': torch.tensor(pair_adj, dtype=torch.float32)
    }

# -------------------------
# Metrics: node accuracy, edge precision/recall/F1
# -------------------------
def node_accuracy(pred_logits, gt_node_types):
    pred = pred_logits.argmax(dim=1)
    return (pred == gt_node_types.to(pred_logits.device)).float().mean().item()

def edge_prf(edge_logits, gt_pair_adj, thr=0.5):
    # flatten excluding diag
    device = edge_logits.device
    N = edge_logits.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    # ensure gt_pair_adj is on same device as mask
    gt_pair_adj = gt_pair_adj.to(device)
    preds = (torch.sigmoid(edge_logits[mask]) >= thr).cpu().numpy().astype(int)
    tgt = gt_pair_adj[mask].cpu().numpy().astype(int)
    # small helper
    from sklearn.metrics import precision_recall_fscore_support
    if preds.sum() == 0 and tgt.sum() == 0:
        return (1.0, 1.0, 1.0)
    p,r,f,_ = precision_recall_fscore_support(tgt, preds, average='binary', zero_division=0)
    return (p,r,f)

# -------------------------
# Training: pretrain contrastive (per-scene, batch_size=1)
# -------------------------
def pretrain_contrastive(model: TypeGraphModel, scenes: List[List[Dict[str,Any]]], device='cpu',
                         epochs=5, lr=1e-3, weight_decay=1e-5):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0; cnt=0
        random.shuffle(scenes)
        for scene in scenes:
            # scene is list of sensor dicts with 'raw' torch tensors (cpu)
            # create two augmentations per sensor
            # move raw to device
            N = len(scene)
            x1 = []
            x2 = []
            for s in scene:
                raw = s['raw']
                a1 = augment_tensor_1d(raw)
                a2 = augment_tensor_1d(raw)
                x1.append(a1.to(device))
                x2.append(a2.to(device))
            # temporarily build pseudo-scenes
            scene1 = [{'id':scene[i]['id'],'type':scene[i]['type'],'raw':x1[i],'meta':scene[i].get('meta',{})} for i in range(N)]
            scene2 = [{'id':scene[i]['id'],'type':scene[i]['type'],'raw':x2[i],'meta':scene[i].get('meta',{})} for i in range(N)]
            out1 = model.forward_scene(scene1)
            out2 = model.forward_scene(scene2)
            z1 = out1['proj']  # (N,E)
            z2 = out2['proj']
            loss = nt_xent_loss(z1, z2, temperature=0.1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total_loss += loss.item(); cnt += 1
        print(f"[Pretrain] Epoch {ep+1}/{epochs} avg_loss={total_loss/max(1,cnt):.6f}")
    return model

# -------------------------
# Finetune hybrid: contrastive + supervised node CE + edge BCE
# batch_size=1 per scene
# -------------------------
def finetune_hybrid(model: TypeGraphModel, scenes_train: List[List[Dict[str,Any]]], scenes_val=None,
                    device='cpu', epochs=8, lr=1e-3, weights=None):
    if weights is None:
        weights = {'node':1.0, 'edge':1.0, 'contrast':0.5}
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        tot=0.0; cnt=0
        random.shuffle(scenes_train)
        for scene in scenes_train:
            gt = build_gt_from_scene(scene)
            gt_node = gt['node_types'].to(device)
            gt_pair = gt['pair_adj'].to(device)
            out = model.forward_scene(scene)
            node_logits = out['node_logits'].to(device)
            edge_logits = out['edge_logits'].to(device)
            proj = out['proj'].to(device)
            # supervised losses
            node_loss = F.cross_entropy(node_logits, gt_node)
            # mask diag
            N = edge_logits.shape[0]
            eye = torch.eye(N, device=device)
            edge_logits_masked = edge_logits * (1 - eye) - 1e9*eye  # avoid self-pair; later we'll replace with masked loss to avoid -1e9 issues
            # safer: compute BCE excluding diag
            mask = (1 - eye).bool()
            edge_logits_no_diag = edge_logits[mask]
            gt_pairs_no_diag = gt_pair[mask]
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits_no_diag, gt_pairs_no_diag)
            # contrastive: create augmentations and compute proj contrastive
            N = len(scene)
            x1 = [augment_tensor_1d(s['raw']).to(device) for s in scene]
            x2 = [augment_tensor_1d(s['raw']).to(device) for s in scene]
            scene1 = [{'id':scene[i]['id'],'type':scene[i]['type'],'raw':x1[i],'meta':scene[i].get('meta',{})} for i in range(N)]
            scene2 = [{'id':scene[i]['id'],'type':scene[i]['type'],'raw':x2[i],'meta':scene[i].get('meta',{})} for i in range(N)]
            z1 = model.forward_scene(scene1)['proj'].to(device)
            z2 = model.forward_scene(scene2)['proj'].to(device)
            contrast_loss = nt_xent_loss(z1, z2, temperature=0.1)
            loss = weights['node']*node_loss + weights['edge']*edge_loss + weights['contrast']*contrast_loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); cnt += 1
        avg = tot/max(1,cnt)
        print(f"[Finetune] Ep {ep+1}/{epochs} train_loss={avg:.6f}")
        # validation
        if scenes_val is not None:
            model.eval()
            with torch.no_grad():
                accs=[]; prs=[]
                for scene in scenes_val:
                    out = model.forward_scene(scene)
                    gt = build_gt_from_scene(scene)
                    accs.append(node_accuracy(out['node_logits'], gt['node_types']))
                    p,r,f = edge_prf(out['edge_logits'], gt['pair_adj'])
                    prs.append((p,r,f))
                print(f"[Val] node_acc={np.mean(accs):.3f} edge_f1={np.mean([x[2] for x in prs]):.3f}")
    return model

# -------------------------
# Demo main
# -------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # build synthetic dataset
    scenes_train = [ synthetic_scene(num_sensors_min=6, num_sensors_max=12, duration_samples=512, seed=i) for i in range(200) ]
    scenes_val = [ synthetic_scene(num_sensors_min=6, num_sensors_max=12, duration_samples=512, seed=1000+i) for i in range(50) ]
    model = TypeGraphModel(emb_dim=128, conv_channels=64, time_pooled=16, transformer_layers=2, nhead=4, device=device)
    print("Pretraining contrastive (small)...")
    pretrain_contrastive(model, scenes_train[:100], device=device, epochs=3, lr=1e-3)
    print("Finetuning hybrid...")
    finetune_hybrid(model, scenes_train, scenes_val, device=device, epochs=6, lr=5e-4)
    print("Done.")
