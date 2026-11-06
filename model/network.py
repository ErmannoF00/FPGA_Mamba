"""Trainable sequence classifier using a bilinear selective state-space layer.

This is a Mamba-derived architecture, not a bitwise reimplementation of
state-spaces/mamba. It replaces the ZOH state update with a bilinear transform
and quantizes time steps to four hardware-addressable values.
"""
import torch
from torch import nn
from torch.nn import functional as F

DIM = 8
STATE = 4
KERNEL = 3
NUM_CLASSES = 10
TIME_STEPS = 8
FRAC = 10
SCALE = 1 << FRAC
LEVELS = (0.125, 0.25, 0.5, 1.0)


def fake_q(x: torch.Tensor) -> torch.Tensor:
    quant = torch.clamp(torch.round(x * SCALE), -32768, 32767) / SCALE
    return x + (quant - x).detach()


class SelectiveClassifier(nn.Module):
    def __init__(self, scheme: str = "bilinear", qat: bool = True):
        super().__init__()
        if scheme not in ("bilinear", "zoh"):
            raise ValueError("scheme must be bilinear or zoh")
        self.scheme = scheme
        self.qat = qat
        self.in_proj = nn.Linear(DIM, DIM)
        self.gate_proj = nn.Linear(DIM, DIM)
        self.conv_weight = nn.Parameter(torch.randn(KERNEL, DIM) * 0.15)
        self.conv_bias = nn.Parameter(torch.zeros(DIM))
        self.b_proj = nn.Linear(DIM, STATE)
        self.c_proj = nn.Linear(DIM, STATE)
        self.dt_proj = nn.Linear(DIM, DIM)
        self.a_logits = nn.Parameter(torch.zeros(DIM, STATE))
        self.skip = nn.Parameter(torch.ones(DIM) * 0.1)
        self.out_proj = nn.Linear(DIM, DIM)
        self.head = nn.Linear(DIM * TIME_STEPS, NUM_CLASSES)
        nn.init.constant_(self.dt_proj.bias, 0.0)

    def q(self, x: torch.Tensor) -> torch.Tensor:
        return fake_q(x) if self.qat else x

    def dynamics(self, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Negative stable poles. Bound their magnitude to keep Tustin alpha positive.
        a = -(0.05 + 1.45 * torch.sigmoid(self.a_logits))
        da = dt.unsqueeze(-1) * a
        if self.scheme == "bilinear":
            denominator = 1 - 0.5 * da
            alpha = (1 + 0.5 * da) / denominator
            gain = dt.unsqueeze(-1) / denominator
        else:
            alpha = torch.exp(da)
            gain = torch.expm1(da) / a
        return self.q(alpha), self.q(gain)

    def forward(self, input_rows: torch.Tensor, state_bits=None) -> torch.Tensor:
        if input_rows.ndim != 3 or tuple(input_rows.shape[1:]) != (TIME_STEPS, DIM):
            raise ValueError("expected (batch, 8, 8)")
        batch = input_rows.shape[0]
        h = input_rows.new_zeros((batch, DIM, STATE))
        history = [input_rows.new_zeros((batch, DIM)) for _ in range(KERNEL - 1)]
        tokens = []
        levels = input_rows.new_tensor(LEVELS)
        thresholds = input_rows.new_tensor((-0.5, 0.0, 0.5))
        for t in range(TIME_STEPS):
            x = self.q(input_rows[:, t, :])
            projected = self.q(self.in_proj(x))
            gated = self.q(self.gate_proj(x))
            taps = [projected] + history
            conv = self.q(sum(self.q(taps[k] * self.q(self.conv_weight[k]))
                              for k in range(KERNEL)) + self.q(self.conv_bias))
            u = self.q(F.relu(conv))
            b = self.q(self.b_proj(u))
            c = self.q(self.c_proj(u))
            log_dt = self.q(self.dt_proj(u))
            idx = torch.bucketize(log_dt.detach(), thresholds)
            dt = levels[idx] + 0.05 * (log_dt - log_dt.detach())
            alpha, gain = self.dynamics(dt)
            excitation = self.q(self.q(gain * b.unsqueeze(1)) * u.unsqueeze(-1))
            h = self.q(self.q(alpha * h) + excitation)
            if state_bits is not None:
                bits = torch.as_tensor(state_bits, dtype=h.dtype, device=h.device)
                if bits.shape != (DIM, STATE):
                    raise ValueError("state_bits must have shape (8,4)")
                step = torch.pow(h.new_tensor(2.0), bits)
                encoded = torch.round(h * step)
                h = torch.minimum(torch.maximum(encoded, -32.0 * step),
                                  32.0 * step - 1.0) / step
            y = self.q(torch.sum(self.q(c.unsqueeze(1) * h), dim=-1)
                       + self.q(self.q(self.skip) * u))
            gate = self.q(torch.clamp((gated + 3.0) / 6.0, 0, 1))
            result = self.q(self.out_proj(self.q(y * gate)))
            tokens.append(self.q(result + projected))
            history = [projected] + history[:-1]
        return self.head(torch.cat(tokens, dim=-1))


def digits_data():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    data = load_digits()
    x = (data.images.astype('float32') / 16.0)
    y = data.target.astype('int64')
    indexes = list(range(len(y)))
    trainval, test = train_test_split(indexes, test_size=0.2, random_state=19, stratify=y)
    train, val = train_test_split(trainval, test_size=0.2, random_state=23,
                                  stratify=y[trainval])
    return ((torch.from_numpy(x[ids]), torch.from_numpy(y[ids]))
            for ids in (train, val, test))
