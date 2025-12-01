# gnn_pretrain_finetune_overflow.py
# Full pipeline: contrastive pretrain -> finetune full GNN model -> evaluation metrics
# Requirements: torch, numpy
# Run: python gnn_pretrain_finetune_overflow.py

import math, random, time
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import multiprocessing

# -------------------------
# Checkpoint utilities
# -------------------------
def save_checkpoint(path, model: nn.Module, optimizer: torch.optim.Optimizer=None, epoch:int=None, extra:dict=None):
    ckpt = {'model_state': model.state_dict()}
    if optimizer is not None:
        ckpt['opt_state'] = optimizer.state_dict()
    if epoch is not None:
        ckpt['epoch'] = int(epoch)
    if extra is not None:
        ckpt['extra'] = extra
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(ckpt, path)
    print(f"[Checkpoint] saved {path}")

def load_checkpoint(path, model: nn.Module, optimizer: torch.optim.Optimizer=None, map_location=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    map_loc = None
    if map_location is not None:
        # accept string like 'cpu'/'cuda' as well
        map_loc = torch.device(map_location) if isinstance(map_location, str) else map_location
    ckpt = torch.load(path, map_location=map_loc)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'opt_state' in ckpt:
        optimizer.load_state_dict(ckpt['opt_state'])
    epoch = ckpt.get('epoch', None)
    extra = ckpt.get('extra', None)
    print(f"[Checkpoint] loaded {path} (epoch={epoch})")
    return epoch, extra

# -------------------------
# Utility / Data generator (simple)
# -------------------------
def example_scene_generator(num_sensors=16, duration_sec=1.0, rates=[200],
                            prob_1word=0.5, prob_2word=0.25, prob_bits=0.25, seed=None):
    """
    Produce a scene list. For 2word, create two node entries with meta['raw32'].
    All sensors generated at rate=200 Hz (unified rate).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    sensors = []
    idx = 0
    for i in range(num_sensors):
        rate = int(np.random.choice(rates))
        T = int(max(8, round(duration_sec * rate)))
        p = np.random.rand()
        if p < prob_1word:
            t = np.linspace(0, duration_sec, T)
            freq = np.random.uniform(0.2, 5.0)
            sig = 2000*np.sin(2*np.pi*freq*t + np.random.rand()*2*np.pi)
            sig += 10000 + np.random.normal(0,50,T)
            sensors.append({'id': idx, 'type':'1word', 'raw_rate':rate, 'raw':sig.astype(np.float32), 'meta':{}})
            idx += 1
        elif p < prob_1word + prob_2word:
            base = np.random.randint(0, 1<<20)
            noise = np.cumsum(np.random.normal(0, 100, T))
            raw32 = (base + noise).astype(np.int64)
            raw32 = np.clip(raw32, 0, (1<<32)-1)
            lsb = (raw32 & 0xFFFF).astype(np.int64)
            msb = ((raw32 >> 16) & 0xFFFF).astype(np.int64)
            sensors.append({'id': idx, 'type':'2word_lsb', 'raw_rate':rate, 'raw':lsb.astype(np.float32),
                            'meta': {'pair': idx+1, 'raw32': raw32}})
            idx += 1
            sensors.append({'id': idx, 'type':'2word_msb', 'raw_rate':rate, 'raw':msb.astype(np.float32),
                            'meta': {'pair': idx-1, 'raw32': raw32}})
            idx += 1
        else:
            bit_width = np.random.randint(1, 8)
            start = np.random.randint(0, 16-bit_width)
            if np.random.rand() < 0.6:
                values = np.cumsum(np.random.choice([0,1], size=T, p=[0.98,0.02])).astype(np.int64)
            else:
                values = (np.random.rand(T) < 0.02).astype(np.int64)
            word = (values & ((1<<bit_width)-1)) << start
            sensors.append({'id': idx, 'type':'bits', 'raw_rate':rate, 'raw':word.astype(np.float32),
                            'meta': {'bit_start':int(start), 'bit_end':int(start+bit_width-1)}})
            idx += 1
    return sensors

# -------------------------
# Scene Dataset
# -------------------------
class SceneDataset(Dataset):
    def __init__(self, scenes: List[List[Dict[str,Any]]]):
        self.scenes = scenes
    def __len__(self):
        return len(self.scenes)
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        # convert to torch now
        scene_copy = []
        for s in scene:
            scene_copy.append({
                'id': s['id'],
                'type': s['type'],
                'raw_rate': s['raw_rate'],
                'raw': torch.from_numpy(np.asarray(s['raw'], dtype=np.float32)),
                'meta': s.get('meta', {})
            })
        return scene_copy

# -------------------------
# Helpers: dataloader factory
# -------------------------
def make_scene_loader(dataset: SceneDataset, batch_size=1, shuffle=False, device='cpu'):
    nw = 0
    pin = False
    if device != 'cpu' and torch.cuda.is_available():
        # GPU available -> allow worker loading + pinned memory
        cpu_cnt = multiprocessing.cpu_count()
        nw = min(4, max(1, cpu_cnt//2))
        pin = True
    # on Windows, lots of workers may be slower; keep moderate
    # use safe collate that returns scene list (works for batch_size=1 and >1)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_identity,
                      num_workers=nw, pin_memory=pin)

# collate: identity (batch_size=1 recommended)
def collate_identity(batch):
    # batch is list of scenes; return single scene when batch_size==1
    if len(batch) == 1:
        return batch[0]
    return batch
# -------------------------
# Augmentations for contrastive
# -------------------------
def augment_time_series_tensor(x: torch.Tensor, jitter_std=0.01, mask_ratio=0.03):
    # x: (T,)
    x = x.clone()
    T = x.shape[0]
    jitter = torch.randn_like(x) * jitter_std * (x.std().item() + 1e-6)
    x = x + jitter
    if mask_ratio > 0:
        mlen = max(1, int(T * mask_ratio))
        start = random.randint(0, max(0, T-mlen))
        x[start:start+mlen] = 0.0
    return x

# NT-Xent
def nt_xent(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = (~torch.eye(2*B, dtype=torch.bool, device=z.device)).float()
    exp_sim = torch.exp(sim) * mask
    denom = exp_sim.sum(dim=1, keepdim=True)
    pos = torch.cat([torch.arange(B,2*B), torch.arange(0,B)]).to(z.device)
    log_prob = sim - torch.log(denom + 1e-12)
    loss = -log_prob[torch.arange(2*B, device=z.device), pos].mean()
    return loss

# -------------------------
# Rate-specific Encoder (outputs: pooled vector + temporal compressed feature)
# -------------------------
class RateEncoderTemporal(nn.Module):
    def __init__(self, emb_dim=128, conv_channels=64, time_bins=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_channels, emb_dim)
        # temporal compressed branch
        self.temporal_pool = nn.AdaptiveAvgPool1d(time_bins)  # compress time -> L bins
        self.temporal_proj = nn.Conv1d(conv_channels, emb_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (T,) or (1, T) or (1,1,T)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1,1,T)
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1,1,T)
        B, C, T = x.shape  # B=1, C=1
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))  # (1, Cc, T)
        pooled = self.pool(h).squeeze(-1)  # (1, Cc)
        vec = self.fc(pooled)  # (1, emb_dim)
        # temporal compressed features
        tcomp = self.temporal_pool(h)  # (1, Cc, L)
        tfeat = self.temporal_proj(tcomp).squeeze(0).permute(1,0)  # (L, emb_dim)
        # return vector (1,emb) and temporal (L, emb)
        return vec.squeeze(0), tfeat  # vec:(emb_dim,), tfeat:(L,emb_dim)

# -------------------------
# Simple Graph Layer (attention)
# -------------------------
class SimpleGraphLayer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.scale = math.sqrt(emb_dim)
    def forward(self, X, att_mask=None):
        Q = self.q(X)
        K = self.k(X)
        V = self.v(X)
        att = torch.matmul(Q, K.T) / self.scale  # (N,N)
        if att_mask is not None:
            att = att + att_mask
        A = F.softmax(att, dim=1)
        agg = torch.matmul(A, V)
        out = F.relu(self.out(agg + X))
        return out, A

# -------------------------
# Full Model (encoders per rate + GNN + heads)
# -------------------------
class SensorStructureModel(nn.Module):
    def __init__(self, rates=[200], emb_dim=128, time_bins=32, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.rates = sorted(rates)
        self.encoders = nn.ModuleDict({str(r): RateEncoderTemporal(emb_dim=emb_dim, time_bins=time_bins) for r in self.rates})
        self.g1 = SimpleGraphLayer(emb_dim)
        self.g2 = SimpleGraphLayer(emb_dim)
        self.node_head = nn.Linear(emb_dim, 3)
        self.edge_head = nn.Sequential(nn.Linear(emb_dim*2, emb_dim), nn.ReLU(), nn.Linear(emb_dim,1))
        self.order_head = nn.Sequential(nn.Linear(emb_dim*2, emb_dim//2), nn.ReLU(), nn.Linear(emb_dim//2,2))
        self.bitmask_head = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.ReLU(), nn.Linear(emb_dim//2,16))
        # overflow temporal predictor: takes concatenated pair temporal features (L, 2*emb) -> conv -> per-bin logits
        self.overflow_conv1 = nn.Conv1d(2*emb_dim, emb_dim, kernel_size=3, padding=1)
        self.overflow_conv2 = nn.Conv1d(emb_dim, emb_dim//2, kernel_size=3, padding=1)
        self.overflow_out = nn.Conv1d(emb_dim//2, 1, kernel_size=1)  # per-bin logits

    def encode_scene(self, scene):
        embs = []
        temps = []  # list of (L,emb)
        metas = []
        types = []
        for s in scene:
            raw_field = s['raw']
            # accept numpy / tensor / list
            if isinstance(raw_field, np.ndarray):
                raw_t = torch.from_numpy(raw_field).float().to(self.device)
            elif isinstance(raw_field, torch.Tensor):
                raw_t = raw_field.float().to(self.device)
            else:
                raw_t = torch.tensor(raw_field, dtype=torch.float32, device=self.device)
            raw_t = raw_t.view(-1)
            rate = int(s['raw_rate'])
            enc = self.encoders[str(rate)]
            vec, tfeat = enc(raw_t)
            embs.append(vec)
            temps.append(tfeat)  # tensor (L,emb)
            metas.append(s.get('meta', {}))
            t = s['type']
            if t == '1word':
                types.append(0)
            elif t.startswith('2word'):
                types.append(1)
            else:
                types.append(2)
        X = torch.stack(embs, dim=0)  # (N,emb)
        return X, temps, metas, torch.tensor(types, dtype=torch.long, device=self.device)

    def forward(self, scene):
        X, temps, metas, types_gt = self.encode_scene(scene)  # X:(N,D) temps: list len N each (L,D)
        H1, A1 = self.g1(X)
        H2, A2 = self.g2(H1)
        node_logits = self.node_head(H2)  # (N,3)
        N = X.shape[0]
        # pair features
        src = H2.unsqueeze(1).expand(N,N,-1)
        dst = H2.unsqueeze(0).expand(N,N,-1)
        pair_feat = torch.cat([src,dst], dim=-1)  # (N,N,2D)
        pair_flat = pair_feat.view(N*N, -1)
        edge_logits = self.edge_head(pair_flat).view(N,N)
        order_logits = self.order_head(pair_flat).view(N,N,2)
        bitmask_logits = self.bitmask_head(H2)  # (N,16)
        # overflow temporal prediction
        # build per-pair temporal concatenation: temps[i] shape (L,D)
        L = temps[0].shape[0]
        # build a (N,N, 2D, L) then conv expects (batch, channels, L) -> we'll iterate pairs (small N)
        overflow_logits = torch.zeros((N,N,L), device=self.device)
        for i in range(N):
            for j in range(N):
                t1 = temps[i].transpose(0,1).unsqueeze(0)  # (1,D,L)
                t2 = temps[j].transpose(0,1).unsqueeze(0)  # (1,D,L)
                pair_t = torch.cat([t1, t2], dim=1)  # (1,2D,L)
                h = F.relu(self.overflow_conv1(pair_t))
                h = F.relu(self.overflow_conv2(h))
                out = self.overflow_out(h).squeeze(0).squeeze(0)  # (L,)
                overflow_logits[i,j] = out
        return {
            'node_logits': node_logits,
            'edge_logits': edge_logits,
            'order_logits': order_logits,
            'bitmask_logits': bitmask_logits,
            'overflow_logits': overflow_logits,  # raw logits per-bin
            'attention': A2
        }

# -------------------------
# Ground truth builder (including overflow event binning)
# -------------------------
def build_ground_truth(scene, time_bins=32):
    N = len(scene)
    node_types = []
    pair_adj = np.zeros((N,N), dtype=np.float32)
    order_label = -np.ones((N,N), dtype=np.int64)
    bitmask = np.zeros((N,16), dtype=np.float32)
    overflow_events_bins = np.zeros((N,N,time_bins), dtype=np.float32)  # per-bin events (0/1)
    overflow_count = np.zeros((N,N), dtype=np.float32)

    # map id->index
    id2idx = {s['id']:i for i,s in enumerate(scene)}
    for i,s in enumerate(scene):
        t = s['type']
        if t == '1word':
            node_types.append(0)
        elif t.startswith('2word'):
            node_types.append(1)
        else:
            node_types.append(2)
        if s['type']=='bits':
            m = s.get('meta',{})
            if 'bit_start' in m:
                bs, be = m['bit_start'], m['bit_end']
                bitmask[i, bs:be+1] = 1.0

    # pair adjacency & overflow event generation
    for i,s in enumerate(scene):
        m = s.get('meta',{})
        if s['type']=='2word_lsb' and 'pair' in m:
            pair_id = m['pair']
            if pair_id in id2idx:
                j = id2idx[pair_id]
                pair_adj[i,j] = 1.0
                pair_adj[j,i] = 1.0
                order_label[i,j] = 1
                order_label[j,i] = 0
                # compute overflow events from raw32 if available in meta of either
                raw32 = m.get('raw32', None)
                if raw32 is None:
                    raw32 = scene[j].get('meta', {}).get('raw32', None)
                if raw32 is not None:
                    raw32 = np.asarray(raw32)
                    lsb = (raw32 & 0xFFFF).astype(np.int64)
                    # event when lsb decreases (mod)
                    events = (lsb[1:] < lsb[:-1]).astype(np.int64)
                    # aggregate to bins
                    L = raw32.shape[0]
                    # create bin indices
                    bins = np.linspace(0, L, time_bins+1, dtype=int)
                    for b in range(time_bins):
                        sidx, eidx = bins[b], bins[b+1]
                        if eidx > sidx:
                            overflow_events_bins[i,j,b] = events[sidx:eidx].sum() > 0  # presence
                            overflow_events_bins[j,i,b] = overflow_events_bins[i,j,b]
                    overflow_count[i,j] = events.sum()
                    overflow_count[j,i] = overflow_count[i,j]
    return {
        'node_types': torch.tensor(node_types, dtype=torch.long),
        'pair_adj': torch.tensor(pair_adj, dtype=torch.float32),
        'order_label': torch.tensor(order_label, dtype=torch.long),
        'bitmask': torch.tensor(bitmask, dtype=torch.float32),
        'overflow_bins': torch.tensor(overflow_events_bins, dtype=torch.float32),
        'overflow_count': torch.tensor(overflow_count, dtype=torch.float32)
    }

# -------------------------
# Metrics
# -------------------------
def metrics_node_acc(node_logits, node_truth):
    pred = torch.argmax(node_logits, dim=1)
    correct = (pred == node_truth).sum().item()
    total = node_truth.numel()
    return correct/total

def metrics_edge_prf(edge_logits, pair_adj, thr=0.5):
    probs = torch.sigmoid(edge_logits).cpu().detach().numpy().ravel()
    tgt = pair_adj.cpu().numpy().ravel()
    pred = (probs >= thr).astype(int)
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(tgt, pred, average='binary', zero_division=0)
    return p, r, f

def metrics_bitmask_iou(bitmask_logits, bitmask_gt, thr=0.5):
    pred = (torch.sigmoid(bitmask_logits) >= thr).cpu().numpy()
    gt = bitmask_gt.cpu().numpy().astype(bool)
    # per-node IoU
    ious = []
    for i in range(gt.shape[0]):
        inter = (pred[i] & gt[i]).sum()
        union = (pred[i] | gt[i]).sum()
        if union==0: iou = 1.0 if inter==0 else 0.0
        else: iou = inter/union
        ious.append(iou)
    return float(np.mean(ious))

def metrics_overflow_event(edge_overflow_logits, overflow_bins_gt, thr=0.0):
    # edge_overflow_logits: (N,N,L) raw logits; gt: (N,N,L) 0/1
    probs = torch.sigmoid(edge_overflow_logits).cpu().detach().numpy()
    gt = overflow_bins_gt.cpu().numpy()
    # flatten across pairs/time but consider only gt pairs that have any events
    preds = (probs >= thr).astype(int).ravel()
    tgs = gt.ravel()
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(tgs, preds, average='binary', zero_division=0)
    return p, r, f

# -------------------------
# Contrastive pretrain (encoders)
# -------------------------
def pretrain_contrastive(encoders: nn.ModuleDict, scenes: List[List[Dict]], device='cpu',
                         epochs=10, batch_size=64, lr=1e-3):
    """
    We'll collect sensor-level items across scenes into a big pool and train in mini-batches.
    For scalability, we sample sensors uniformly from scenes per batch.
    """
    # build pool of (raw, rate_key)
    pool = []
    for sc in scenes:
        for s in sc:
            pool.append((s['raw'], s['raw_rate']))
    print(f"[Pretrain] Pool size: {len(pool)} sensors")

    # ensure encoders are on device (use torch.device for safety)
    dev = torch.device(device)
    encoders.to(dev)

    opt = torch.optim.Adam(encoders.parameters(), lr=lr)
    for ep in range(epochs):
        random.shuffle(pool)
        total_loss = 0.0
        cnt = 0
        for i in range(0, len(pool), batch_size):
            batch = pool[i:i+batch_size]
            # build two augmented views per item, encode with corresponding encoder
            vecs1 = []
            vecs2 = []
            for raw, rate in batch:
                # safe conversion: accept numpy arrays or torch tensors or lists
                if isinstance(raw, np.ndarray):
                    raw_t = torch.from_numpy(raw).float().to(dev)
                elif isinstance(raw, torch.Tensor):
                    raw_t = raw.float().to(dev)
                else:
                    raw_t = torch.tensor(raw, dtype=torch.float32, device=dev)
                # ensure 1D tensor
                raw_t = raw_t.view(-1)
                aug1 = augment_time_series_tensor(raw_t)
                aug2 = augment_time_series_tensor(raw_t)
                enc = encoders[str(int(rate))]
                v1, _ = enc(aug1)
                v2, _ = enc(aug2)
                vecs1.append(v1.unsqueeze(0))
                vecs2.append(v2.unsqueeze(0))
            if len(vecs1) == 0:
                continue
            z1 = torch.cat(vecs1, dim=0)
            z2 = torch.cat(vecs2, dim=0)
            loss = nt_xent(z1, z2, temperature=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); cnt += 1
        print(f"[Pretrain] Epoch {ep+1}/{epochs} avg_loss={total_loss/max(1,cnt):.6f}")
    return encoders

# -------------------------
# Full training (finetune)
# -------------------------
def finetune_full(model: SensorStructureModel, train_ds: SceneDataset, val_ds: SceneDataset=None,
                  device='cpu', epochs=10, lr=1e-3, weights=None):
    if weights is None:
        weights = {'node':1.0, 'edge':1.0, 'order':0.5, 'bitmask':1.0, 'overflow_bin':1.0, 'overflow_cnt':0.1}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = make_scene_loader(train_ds, batch_size=1, shuffle=True, device=device)
    val_loader = None if val_ds is None else make_scene_loader(val_ds, batch_size=1, shuffle=False, device=device)

    model = model.to(device)
    use_amp = (device != 'cpu') and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # enable cudnn benchmark for fixed-size convs (may help performance on GPU)
    if device != 'cpu' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    max_grad_norm = 5.0  # gradient clipping
    for ep in range(epochs):
        model.train()
        tot_loss=0.0
        cnt=0
        for scene in train_loader:
            scene_list = scene
            # build s_for_gt with numpy raw arrays (build_ground_truth expects numpy arrays)
            s_for_gt = []
            for s in scene_list:
                raw_field = s['raw']
                if isinstance(raw_field, torch.Tensor):
                    # allow non_blocking transfer when pinned memory used
                    raw_np = raw_field.detach().cpu().numpy()
                else:
                    raw_np = np.asarray(raw_field)
                s_for_gt.append({'id': s['id'], 'type': s['type'], 'raw_rate': s['raw_rate'], 'raw': raw_np, 'meta': s.get('meta', {})})
            gt = build_ground_truth(s_for_gt)
            # move gt tensors to device
            for k in list(gt.keys()):
                if isinstance(gt[k], torch.Tensor):
                    gt[k] = gt[k].to(device)
            opt.zero_grad()
            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(scene_list)
                    # sanitize model outputs to remove NaN/Inf
                    for k, v in list(out.items()):
                        if isinstance(v, torch.Tensor):
                            out[k] = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)

                    # sanitize GT adjacency/targets
                    pair_adj = torch.nan_to_num(gt['pair_adj'].to(device), nan=0.0, posinf=1.0, neginf=0.0)

                    # node loss
                    node_logits = out['node_logits']
                    node_loss = F.cross_entropy(node_logits, gt['node_types'].to(device))

                    # edge loss: BCEWithLogits, mask diagonal out safely
                    edge_logits = out['edge_logits']
                    N = edge_logits.shape[0]
                    eye = torch.eye(N, device=device)
                    edge_loss_per = F.binary_cross_entropy_with_logits(edge_logits, pair_adj, reduction='none')
                    if N > 1:
                        edge_loss = (edge_loss_per * (1.0 - eye)).sum() / float(N * N - N)
                    else:
                        edge_loss = edge_loss_per.mean()

                    # order loss
                    order_label = gt['order_label'].to(device)
                    mask = (order_label != -1)
                    if mask.any():
                        pred_order = out['order_logits'].view(-1, 2)
                        true_order = order_label.view(-1)
                        m = mask.view(-1)
                        order_loss = F.cross_entropy(pred_order[m], true_order[m])
                    else:
                        order_loss = torch.tensor(0.0, device=device)

                    # bitmask loss (use logits + BCEWithLogits)
                    is_bits = (gt['node_types'].to(device) == 2)
                    if is_bits.any():
                        bitmask_loss = F.binary_cross_entropy_with_logits(
                            out['bitmask_logits'][is_bits],
                            gt['bitmask'].to(device)[is_bits]
                        )
                    else:
                        bitmask_loss = torch.tensor(0.0, device=device)

                    # overflow losses
                    overflow_logits = out['overflow_logits']
                    overflow_logits = torch.nan_to_num(overflow_logits, nan=0.0, posinf=1e6, neginf=-1e6)
                    overflow_bins = gt['overflow_bins'].to(device)
                    overflow_bin_loss = F.binary_cross_entropy_with_logits(overflow_logits, overflow_bins)
                    overflow_cnt_pred = torch.sigmoid(overflow_logits).sum(dim=2)
                    overflow_cnt_loss = F.mse_loss(overflow_cnt_pred, gt['overflow_count'].to(device))

                    loss = (weights['node']*node_loss + weights['edge']*edge_loss + weights['order']*order_loss +
                            weights['bitmask']*bitmask_loss + weights['overflow_bin']*overflow_bin_loss +
                            weights['overflow_cnt']*overflow_cnt_loss)

                # ensure finite before backward
                if not torch.isfinite(loss):
                    print(f"[Warning] non-finite loss at epoch {ep+1}, skipping step. loss={loss}")
                    opt.zero_grad()
                    continue

                if use_amp:
                    scaler.scale(loss).backward()
                    # unscale before clipping
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()

                # accumulate only finite scalar
                lval = float(loss.detach().cpu().item())
                if math.isfinite(lval):
                    tot_loss += lval
                    cnt += 1
                else:
                    print(f"[Warning] loss.item() not finite after step; skipping accumulation. raw={lval}")

            except RuntimeError as e:
                print(f"[Error] RuntimeError during training step: {e}. Skipping batch.")
                opt.zero_grad()
                continue

        avg_loss = tot_loss / max(1, cnt) if cnt > 0 else float('nan')
        print(f"[Finetune] Ep {ep+1}/{epochs} train_loss={avg_loss:.6f} (updated_steps={cnt})")

        # Validation metrics (unchanged semantics)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                nod_accs=[]; edge_prfs=[]; bit_iou=[]; over_prfs=[]
                for scene in val_loader:
                    s_for_gt=[]
                    for s in scene:
                        raw_field = s['raw']
                        if isinstance(raw_field, torch.Tensor):
                            raw_np = raw_field.detach().cpu().numpy()
                        else:
                            raw_np = np.asarray(raw_field)
                        s_for_gt.append({'id': s['id'], 'type': s['type'], 'raw_rate': s['raw_rate'], 'raw': raw_np, 'meta': s.get('meta', {})})
                    gt = build_ground_truth(s_for_gt)
                    for k in list(gt.keys()):
                        if isinstance(gt[k], torch.Tensor):
                            gt[k] = gt[k].to(device)
                    out = model(scene)
                    nod_accs.append(metrics_node_acc(out['node_logits'], gt['node_types']))
                    try:
                        p,r,f = metrics_edge_prf(out['edge_logits'], gt['pair_adj'])
                        edge_prfs.append((p,r,f))
                    except:
                        edge_prfs.append((0,0,0))
                    bit_iou.append(metrics_bitmask_iou(out['bitmask_logits'], gt['bitmask']))
                    try:
                        p,r,f = metrics_overflow_event(out['overflow_logits'], gt['overflow_bins'])
                        over_prfs.append((p,r,f))
                    except:
                        over_prfs.append((0,0,0))
                print(f"[Val] node_acc={np.mean(nod_accs):.3f} edge_prf_mean={np.mean([x[2] for x in edge_prfs]):.3f} bit_iou={np.mean(bit_iou):.3f} overflow_f1={np.mean([x[2] for x in over_prfs]):.3f}")
    return model

def evaluate_model(model: SensorStructureModel, dataset: SceneDataset, device='cpu', max_scenes=None):
    """
    Run model on scenes from dataset and compare to ground truth metrics.
    Returns aggregated metrics dict.
    """
    loader = make_scene_loader(dataset, batch_size=1, shuffle=False, device=device)
    model = model.to(device)
    model.eval()
    nod_accs = []
    edge_prfs = []
    bit_ious = []
    over_prfs = []
    with torch.no_grad():
        for idx, scene in enumerate(loader):
            if max_scenes is not None and idx >= max_scenes:
                break
            # build ground-truth input (numpy raws) as build_ground_truth expects
            s_for_gt = []
            for s in scene:
                raw_field = s['raw']
                if isinstance(raw_field, torch.Tensor):
                    raw_np = raw_field.detach().cpu().numpy()
                else:
                    raw_np = np.asarray(raw_field)
                s_for_gt.append({'id': s['id'], 'type': s['type'], 'raw_rate': s['raw_rate'], 'raw': raw_np, 'meta': s.get('meta', {})})
            gt = build_ground_truth(s_for_gt)
            # move gt tensors to device for loss/metrics that expect same device (metrics convert to cpu internally)
            for k in list(gt.keys()):
                if isinstance(gt[k], torch.Tensor):
                    gt[k] = gt[k].to(device)
            out = model(scene)
            # node acc
            nod_accs.append(metrics_node_acc(out['node_logits'], gt['node_types']))
            # edge prf
            try:
                p, r, f = metrics_edge_prf(out['edge_logits'], gt['pair_adj'])
                edge_prfs.append((p, r, f))
            except Exception:
                edge_prfs.append((0.0, 0.0, 0.0))
            # bitmask IoU
            bit_ious.append(metrics_bitmask_iou(out['bitmask_logits'], gt['bitmask']))
            # overflow event prf
            try:
                p, r, f = metrics_overflow_event(out['overflow_logits'], gt['overflow_bins'])
                over_prfs.append((p, r, f))
            except Exception:
                over_prfs.append((0.0, 0.0, 0.0))
    n = len(nod_accs) if len(nod_accs)>0 else 1
    mean_node = float(np.mean(nod_accs)) if len(nod_accs)>0 else 0.0
    mean_edge_f1 = float(np.mean([x[2] for x in edge_prfs])) if len(edge_prfs)>0 else 0.0
    mean_bit_iou = float(np.mean(bit_ious)) if len(bit_ious)>0 else 0.0
    mean_over_f1 = float(np.mean([x[2] for x in over_prfs])) if len(over_prfs)>0 else 0.0
    print(f"[Eval] scenes={len(nod_accs)} node_acc={mean_node:.4f} edge_f1={mean_edge_f1:.4f} bit_iou={mean_bit_iou:.4f} overflow_f1={mean_over_f1:.4f}")
    return {'node_acc': mean_node, 'edge_f1': mean_edge_f1, 'bit_iou': mean_bit_iou, 'overflow_f1': mean_over_f1}

# -------------------------
# Demo / main
# -------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | CPU cores: {multiprocessing.cpu_count()} | RAM: ~24GB reported")
    # generate scenes
    scenes_train = [ example_scene_generator(num_sensors=12, duration_sec=1.0, seed=i) for i in range(100) ]
    scenes_val = [ example_scene_generator(num_sensors=12, duration_sec=1.0, seed=100+i) for i in range(20) ]
    # Pretrain: collect encoders dict
    model = SensorStructureModel(rates=[200], emb_dim=128, time_bins=32, device=device)

    # Pretrain encoders
    print("=== Contrastive pretrain encoders ===")
    encoders = model.encoders
    pretrain_contrastive(encoders, scenes_train, device=device, epochs=2, batch_size=128, lr=1e-3)

    num_epochs = 6
    # Fine-tune full model
    train_ds = SceneDataset(scenes_train)
    val_ds = SceneDataset(scenes_val)
    print("=== Finetune full model ===")
    finetune_full(model, train_ds, val_ds, device=device, epochs=num_epochs, lr=1e-3)
    # Save checkpoint
    ckpt_path = "checkpoints/model_ckpt.pth"
    save_checkpoint(ckpt_path, model, optimizer=None, epoch=num_epochs, extra={'rates': model.rates})
    # Evaluate and print
    print("=== Evaluation on training set ===")
    res_train = evaluate_model(model, train_ds, device=device)
    print("=== Evaluation on validation set ===")
    res_val = evaluate_model(model, val_ds, device=device)
    # Load into fresh model and continue training for comparison
    print("=== Reload checkpoint and continue training (compare) ===")
    model2 = SensorStructureModel(rates=[200], emb_dim=128, time_bins=32, device=device)
    load_checkpoint(ckpt_path, model2, map_location=device)
    # continue training few more epochs
    finetune_full(model2, train_ds, val_ds, device=device, epochs=3, lr=5e-4)
    print("=== Post-resume evaluation on validation set ===")
    res_val2 = evaluate_model(model2, val_ds, device=device)
    print("Summary: val before resume:", res_val, " val after resume:", res_val2)
    
    # ===== PRE-CLASSIFIER INTEGRATION =====
    print("\n" + "="*80)
    print("=== PRE-CLASSIFIER: SENSOR TYPE CLASSIFICATION ===")
    print("="*80)
    
    # Import pre_classifier
    try:
        from pre_classifier import SensorPreClassifier
        
        print("\n[Pre-classifier] Initializing...")
        pre_classifier = SensorPreClassifier(use_deep_learning=True, device=device)
        
        # Convert scenes to pre_classifier format
        def convert_scenes_format(scenes_torch):
            converted = []
            for scene_torch in scenes_torch:
                scene_converted = []
                for s in scene_torch:
                    raw_field = s['raw']
                    if isinstance(raw_field, torch.Tensor):
                        raw_np = raw_field.detach().cpu().numpy()
                    else:
                        raw_np = np.asarray(raw_field)
                    
                    sensor_dict = {
                        'id': s['id'],
                        'type': s['type'],
                        'raw_rate': s['raw_rate'],
                        'raw': raw_np.astype(np.float32),
                        'meta': s.get('meta', {})
                    }
                    scene_converted.append(sensor_dict)
                converted.append(scene_converted)
            return converted
        
        # Convert train/val scenes
        scenes_train_for_preclassifier = convert_scenes_format(scenes_train)
        scenes_val_for_preclassifier = convert_scenes_format(scenes_val)
        
        # Train pre_classifier
        print("\n[Pre-classifier] Training on train set...")
        pre_classifier.train_on_scenes(scenes_train_for_preclassifier, device=device, 
                                       epochs=10, lr=1e-3, batch_size=64)
        
        # Validate pre_classifier
        print("\n[Pre-classifier] Validating on validation set...")
        preclassifier_metrics = pre_classifier.validate_on_scenes(
            scenes_val_for_preclassifier, device=device
        )
        
        print("\n[Pre-classifier] Validation Results:")
        for key, value in preclassifier_metrics.items():
            print(f"  {key:20s}: {value:.4f}")
        
        # Demo on sample scenes
        print("\n[Pre-classifier] Demo Classification on Sample Sensors:")
        demo_count = 0
        for scene_idx, scene in enumerate(scenes_val_for_preclassifier[:2]):
            print(f"\n  Scene {scene_idx}:")
            for sensor_idx, sensor in enumerate(scene[:5]):
                signal = sensor['raw']
                true_type = sensor['type']
                
                result = pre_classifier.classify_signal(signal)
                pred_type = result['type']
                confidence = result['confidence']
                
                match = "✓" if pred_type in true_type or true_type in pred_type else "✗"
                print(f"    {match} Sensor {sensor_idx}: True={true_type:15s} " +
                      f"Pred={pred_type:12s} Conf={confidence:.3f} Method={result['method']}")
                
                demo_count += 1
                if demo_count >= 15:
                    break
            if demo_count >= 15:
                break
        
        print("\n[Pre-classifier] Summary:")
        print(f"  - Accuracy: {preclassifier_metrics.get('accuracy', 0.0):.4f}")
        print(f"  - Successfully classified {len(scenes_val_for_preclassifier)} validation scenes")
        print(f"  - Classification methods: heuristic + deep learning (hybrid)")
        
    except ImportError as e:
        print(f"[Warning] Could not import pre_classifier: {e}")
        print("         Skipping pre-classifier integration.")
    except Exception as e:
        print(f"[Error] Pre-classifier integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Quick hardware-aware tips
    print("\nHardware tips:")
    print(" - Use device='cuda' (Titan X) for heavy ops; enable mixed-precision (autocast used).")
    print(" - DataLoader uses pin_memory + num_workers when GPU available.")
    print(" - If memory allows, increase scene generation parallelism or batch size.")
    print(" - For CPU-bound preprocessing consider increasing num_workers or pre-saving tensors.")
    print(" - If training still slow: reduce emb_dim, reduce time_bins, or reduce dataset size for experiments.")
    print("Done.")