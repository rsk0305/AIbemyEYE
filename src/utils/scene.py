
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import multiprocessing
import random

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
