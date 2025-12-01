'''
모델 관련된 유틸
'''

import torch
import torch.nn as nn

import os
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
