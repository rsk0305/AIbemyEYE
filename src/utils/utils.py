from typing import List, Dict, Any
import numpy as np
import random

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import multiprocessing


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