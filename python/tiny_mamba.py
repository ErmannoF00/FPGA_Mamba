import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os

EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Q15 = 2**15
os.makedirs("weights", exist_ok=True)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MambaBlock(nn.Module):
    def __init__(self, dim=64, n_state=16, dt_rank=4):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.x_proj = nn.Linear(dim, dt_rank + 2 * n_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, dim)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_state + 1.).repeat(dim, 1)))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        x1, x2 = self.in_proj(x).chunk(2, dim=-1)
        x1 = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)
        x1 = F.gelu(x1)
        x2 = F.gelu(x2)
        ssm_out = self.run_ssm(x1)
        return self.out_proj(ssm_out * x2)

    def run_ssm(self, x):
        B, L, D = x.shape
        N = self.A_log.shape[1]
        A = -torch.exp(self.A_log)
        D_vec = self.D

        x_proj = self.x_proj(x)
        dt_rank = self.dt_proj.in_features
        delta, B_, C_ = x_proj.split([dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bln->bldn", delta, B_)

        x_state = torch.zeros((B, D, N), device=x.device)
        ys = []
        for t in range(L):
            x_state = deltaA[:, t] * x_state + deltaB_u[:, t]
            yt = torch.einsum("bdn,bn->bd", x_state, C_[:, t, :])
            ys.append(yt)

        y = torch.stack(ys, dim=1)
        return y + x * D_vec.unsqueeze(0).unsqueeze(0)

class MambaClassifier(nn.Module):
    def __init__(self, dim=64, depth=3):
        super().__init__()
        self.proj = nn.Linear(28, dim)
        self.blocks = nn.Sequential(*[MambaBlock(dim=dim) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.head(x)

def train_and_export():
    model = MambaClassifier().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            opt.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, label)
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def to_q15(t):
        arr = t.detach().cpu().numpy().flatten()
        return np.clip(np.round(arr * Q15), -32768, 32767).astype(np.int16)

    def dump(arr, fname):
        with open(fname, 'w') as f:
            for val in arr:
                f.write(f"{int(val) & 0xFFFF:016b}\n")

    dump(to_q15(model.head.weight), "weights/head_weight.mem")
    dump(to_q15(model.head.bias),   "weights/head_bias.mem")
    print("Export complete. Check ./weights/*.mem")

if __name__ == "__main__":
    train_and_export()
