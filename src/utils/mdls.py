'''
모델 관련된 유틸
'''

import torch
import torch.nn as nn

import os

from datetime import datetime


# -------------------------
# Checkpoint utilities
# -------------------------

def get_timestamp():

    # 1. 현재 날짜와 시간 가져오기
    now = datetime.now()

    # 2. 원하는 형식(YYMMDDHHMM)으로 포맷팅
    # %y: 연도 (두 자리), %m: 월, %d: 일, %H: 시 (24시간제), %M: 분
    return now.strftime("%y%m%d%H%M") # 예: '2512020811' (25년 12월 2일 08시 11분)


def save_checkpoint(path, model: nn.Module, optimizer: torch.optim.Optimizer=None, epoch:int=None, extra:dict=None):
    ckpt = {'model_state': model.state_dict()}
    if optimizer is not None:
        ckpt['opt_state'] = optimizer.state_dict()
    if epoch is not None:
        ckpt['epoch'] = int(epoch)
    if extra is not None:
        ckpt['extra'] = extra
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    timestamp = get_timestamp()
    name, extension = os.path.splitext(path)
    new_filename = f"{name}_{timestamp}{extension}"
    torch.save(ckpt, new_filename)
    print(f"[Checkpoint] saved {new_filename})")

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
