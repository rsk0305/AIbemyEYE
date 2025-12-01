

import torch
import torch.nn.functional as F

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