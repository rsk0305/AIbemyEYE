# unsupervised_clustering_time_series.py
# Skeleton: Pretrain (contrastive) -> Embedding extraction -> KMeans init -> DEC fine-tune -> Predict Nx1 labels

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

# -----------------------------
# Dataset
# -----------------------------
class TimeSeriesDataset(Dataset):
    """Input: data: (N, T) numpy or torch tensor
       We treat each row as one 'instance' (sensor)."""
    def __init__(self, data, augment_fn=None):
        # data: numpy array (N, T) or torch tensor
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()
        self.augment_fn = augment_fn

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]  # (T,)
        if self.augment_fn is None:
            return x.unsqueeze(0)  # (1, T) as channel-first
        else:
            return self.augment_fn(x).unsqueeze(0)

# -----------------------------
# Augmentations for contrastive
# -----------------------------
def augment_time_series(x, jitter_scale=0.01, mask_ratio=0.1, flip_prob=0.0):
    # x: torch tensor (T,)
    x = x.clone()
    # jitter
    x = x + jitter_scale * torch.randn_like(x)
    # random mask segments
    T = x.shape[0]
    if mask_ratio > 0:
        mask_len = int(T * mask_ratio)
        start = np.random.randint(0, max(1, T - mask_len + 1))
        x[start:start+mask_len] = 0.0
    # optional bitflip / discrete augment not included - adapt for bitfields
    return x

def two_view_augment(x):
    a = augment_time_series(x, jitter_scale=0.01, mask_ratio=0.05)
    b = augment_time_series(x, jitter_scale=0.02, mask_ratio=0.07)
    return a.unsqueeze(0), b.unsqueeze(0)  # (1,T), (1,T)

# -----------------------------
# Encoder: 1D-CNN + Temporal Attention Pooling
# -----------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, in_ch=1, conv_channels=64, kernel_size=7, num_layers=3, emb_dim=128):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_layers):
            layers.append(nn.Conv1d(ch, conv_channels, kernel_size, padding=kernel_size//2, stride=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(conv_channels))
            ch = conv_channels
        self.conv = nn.Sequential(*layers)
        # attention pooling
        self.attn = nn.Sequential(nn.Linear(conv_channels, 64), nn.Tanh(), nn.Linear(64, 1))
        self.proj = nn.Linear(conv_channels, emb_dim)

    def forward(self, x):
        # x: (B, 1, T)
        h = self.conv(x)  # (B, C, T)
        # attention across time
        # permute to (B, T, C)
        h_t = h.permute(0, 2, 1)
        attn_w = self.attn(h_t)  # (B, T, 1)
        attn_w = F.softmax(attn_w, dim=1)  # (B, T, 1)
        pooled = (attn_w * h_t).sum(dim=1)  # (B, C)
        z = self.proj(pooled)  # (B, emb_dim)
        z = F.normalize(z, dim=1)
        return z

# -----------------------------
# Contrastive (NT-Xent) loss
# -----------------------------
def nt_xent_loss(z1, z2, temperature=0.1):
    # z1,z2: (B, D)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # 2B x D
    sim = torch.matmul(z, z.T) / temperature
    # mask self
    mask = (~torch.eye(2*batch_size, dtype=torch.bool, device=z.device)).float()
    exp_sim = torch.exp(sim) * mask
    denom = exp_sim.sum(dim=1, keepdim=True)
    # positive indices
    pos = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(z.device)
    log_prob = sim - torch.log(denom + 1e-8)
    loss = -log_prob[torch.arange(2*batch_size).to(z.device), pos].mean()
    return loss

# -----------------------------
# DEC clustering head (Student t-distribution)
# -----------------------------
def student_t_distribution(z, cluster_centers, alpha=1.0):
    # z: (N, D), cluster_centers: (K, D)
    # returns q_ij soft assignments (N, K)
    # q_ij ~ (1 + dist^2/alpha)^{- (alpha+1)/2}
    # We'll compute pairwise squared euclidean
    N = z.shape[0]
    K = cluster_centers.shape[0]
    # expand
    z_expand = z.unsqueeze(1)  # (N,1,D)
    c_expand = cluster_centers.unsqueeze(0)  # (1,K,D)
    dist2 = torch.sum((z_expand - c_expand)**2, dim=2)  # (N,K)
    num = (1.0 + dist2 / alpha) ** (-(alpha + 1) / 2)
    q = num / (num.sum(dim=1, keepdim=True) + 1e-12)
    return q

def target_distribution(q):
    # DEC target distribution p
    weight = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-12)
    p = (weight.t() / weight.sum(dim=1)).t()
    return p

# -----------------------------
# Full training skeleton
# -----------------------------
class UnsupervisedClustering:
    def __init__(self, data_np, device='cpu', pretrain_epochs=50, batch_size=64, emb_dim=128, n_clusters=3):
        """
        data_np: numpy array (N, T)
        """
        self.device = torch.device(device)
        self.data = data_np
        self.N, self.T = data_np.shape
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.n_clusters = n_clusters

        self.encoder = TemporalEncoder(in_ch=1, conv_channels=64, num_layers=3, emb_dim=emb_dim).to(self.device)

        # dataloaders
        ds = TimeSeriesDataset(torch.from_numpy(data_np).float(), augment_fn=None)
        self.dl_all = DataLoader(ds, batch_size=batch_size, shuffle=False)
        # for contrastive pretrain we create a special dataset that yields two-views
        self.ds_aug = TimeSeriesDataset(torch.from_numpy(data_np).float(), augment_fn=None)
        self.dl_aug = DataLoader(self.ds_aug, batch_size=batch_size, shuffle=True)

        # placeholders for cluster centers (torch tensor)
        self.cluster_centers = None

        # optimizer
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3, weight_decay=1e-5)

    def pretrain_contrastive(self, epochs=50):
        self.encoder.train()
        for ep in range(epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(self.dl_aug):
                # batch: (B, 1, T) - but augment_fn is None; we make two views manually
                x = batch.squeeze(1).to(self.device)  # (B, T)
                # create two views
                v1 = []
                v2 = []
                for i in range(x.shape[0]):
                    a, b = two_view_augment(x[i])
                    v1.append(a)
                    v2.append(b)
                v1 = torch.stack(v1, dim=0).to(self.device)  # (B,1,T)
                v2 = torch.stack(v2, dim=0).to(self.device)
                z1 = self.encoder(v1)
                z2 = self.encoder(v2)
                loss = nt_xent_loss(z1, z2, temperature=0.1)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            # print progress
            if (ep + 1) % max(1, epochs // 5) == 0:
                print(f"[Pretrain] Epoch {ep+1}/{epochs} loss={total_loss/len(self.dl_aug):.4f}")

    def extract_embeddings(self):
        self.encoder.eval()
        zs = []
        with torch.no_grad():
            for batch in self.dl_all:
                x = batch.squeeze(1).to(self.device)
                z = self.encoder(x.unsqueeze(1))  # (B, emb_dim)
                zs.append(z.cpu())
        Z = torch.cat(zs, dim=0)  # (N, emb_dim)
        return Z

    def init_kmeans(self, Z):
        km = KMeans(n_clusters=self.n_clusters, n_init=20)
        y = km.fit_predict(Z.numpy())
        centers = torch.from_numpy(km.cluster_centers_).float().to(self.device)
        self.cluster_centers = centers  # (K, D)
        return y

    def dec_finetune(self, max_iter=500, update_interval=10, tol=1e-3, lr=1e-4):
        # DEC-style refinement
        # prepare optimizer for encoder + cluster centers
        # cluster_centers as parameter
        centers = nn.Parameter(self.cluster_centers.clone())
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + [centers], lr=lr)
        self.encoder.train()
        Z = self.extract_embeddings().to(self.device)  # initial embeddings
        q = student_t_distribution(Z, centers.data)
        p = target_distribution(q)
        prev_q = q.clone()
        for it in range(max_iter):
            # compute current embeddings in batches and q
            Z = self.extract_embeddings().to(self.device)
            q = student_t_distribution(Z, centers)
            p = target_distribution(q)

            # convert p to tensor for batch training
            # train small number of gradient steps using full dataset (can be mini-batched)
            # We'll do one epoch over full dataset with KL loss between q and p
            total_loss = 0.0
            # create dataloader that yields indices to compute per-instance q
            for batch in self.dl_all:
                x = batch.squeeze(1).to(self.device)
                z_batch = self.encoder(x.unsqueeze(1))  # (B, D)
                # compute q_batch with current centers
                q_batch = student_t_distribution(z_batch, centers)
                # get corresponding p values by indexing (we need the same ordering)
                # hack: compute indices for this batch from pointer (we'll assume DataLoader not shuffled)
                # To keep skeleton simple, compute p_batch by recomputing p for these z_batch positions using distances
                p_batch = target_distribution(student_t_distribution(z_batch.detach(), centers))
                # KL divergence KL(p || q)
                loss = F.kl_div(q_batch.log(), p_batch, reduction='batchmean')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (it + 1) % update_interval == 0:
                # compute change in q to check convergence
                Z = self.extract_embeddings().to(self.device)
                q_new = student_t_distribution(Z, centers.detach())
                delta = torch.sum(torch.abs(q_new - prev_q))
                prev_q = q_new
                print(f"[DEC] iter {it+1} loss_epoch={total_loss/len(self.dl_all):.4f} delta_q={delta:.6f}")
                if delta < tol:
                    break
        # after fine-tune, set cluster centers
        self.cluster_centers = centers.detach()
        print("DEC fine-tune finished.")

    def predict(self):
        Z = self.extract_embeddings().to(self.device)
        q = student_t_distribution(Z, self.cluster_centers)
        labels = torch.argmax(q, dim=1).cpu().numpy()
        return labels

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Suppose data_np is (N, T) numpy array
    N = 200
    T = 200
    # generate toy random data for demo (replace with your generated dataset)
    rng = np.random.RandomState(0)
    # e.g. cluster 0: sin, cluster1: step, cluster2: sparse bits
    data = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        c = i % 3
        if c == 0:
            t = np.arange(T) / T
            data[i] = 1000 * np.sin(2 * np.pi * 5 * t) + 50 * rng.randn(T)
        elif c == 1:
            data[i] = np.linspace(0, 3000, T) + 50 * rng.randn(T)
        else:
            # sparse bits: mostly zeros with occasional spikes
            data[i] = (rng.rand(T) < 0.03).astype(float) * 60000

    clim = UnsupervisedClustering(data_np=data, device='cpu', pretrain_epochs=10, batch_size=32, emb_dim=128, n_clusters=3)
    print("Pretraining contrastive...")
    clim.pretrain_contrastive(epochs=10)
    print("Extract embeddings...")
    Z = clim.extract_embeddings()
    print("Init KMeans...")
    init_labels = clim.init_kmeans(Z)
    print("Do DEC fine-tune...")
    clim.dec_finetune(max_iter=200, update_interval=10)
    print("Predict final labels...")
    labels = clim.predict()
    print("labels shape:", labels.shape)
