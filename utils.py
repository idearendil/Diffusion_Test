# utils.py
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN = 10


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Data loading
# =========================
def list_tickers(split_dir: Path) -> List[str]:
    xs = sorted(split_dir.glob("*_x.pt"))
    tickers = [p.name[:-5] for p in xs]  # remove "_x.pt"
    return tickers


def load_split_tensors(split_dir: Path, tickers: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: [T, N, F] float32
      Y: [T, N] float32
    """
    x_list = []
    y_list = []
    for tkr in tickers:
        x_path = split_dir / f"{tkr}_x.pt"
        y_path = split_dir / f"{tkr}_y.pt"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing tensor for ticker={tkr} in {split_dir}")

        x = torch.load(x_path, map_location="cpu")  # [T, F]
        y = torch.load(y_path, map_location="cpu")  # [T]
        if x.ndim != 2 or y.ndim != 1:
            raise ValueError(f"Bad shape: {tkr}: x={tuple(x.shape)} y={tuple(y.shape)}")

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Length mismatch: {tkr}: x_T={x.shape[0]} y_T={y.shape[0]}")

        x_list.append(x.float())
        y_list.append(y.float())

    # stack: N tickers
    # X: [N, T, F] -> [T, N, F]
    X = torch.stack(x_list, dim=0).transpose(0, 1).contiguous()
    Y = torch.stack(y_list, dim=0).transpose(0, 1).contiguous()
    return X, Y


class TimeIndexDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        """
        X: [T, N, F]
        Y: [T, N]
        """
        assert X.ndim == 3 and Y.ndim == 2
        assert X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1]
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0] - SEQ_LEN + 1

    def __getitem__(self, idx: int):
        return self.X[idx:idx+SEQ_LEN], self.Y[idx+SEQ_LEN-1]


# =========================
# Diffusion schedules / helpers
# =========================
def cosine_beta_schedule(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """
    Cosine schedule (Nichol & Dhariwal).
    Returns betas: shape [T], float32
    - s: small offset (paper에서 0.008 추천)
    - max_beta: beta upper cap (수치 안정용)
    """
    steps = torch.arange(T + 1, dtype=torch.float64)  # 0..T
    t = steps / T

    # alpha_bar(t) = cos^2( (t+s)/(1+s) * pi/2 )
    alphas_bar = torch.cos(((t + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]  # normalize so alpha_bar(0)=1

    # betas_t = 1 - alpha_bar(t+1) / alpha_bar(t)
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = torch.clamp(betas, min=1e-8, max=max_beta).float()  # [T]
    return betas


def make_t_seq(T: int, sample_steps: int, device: torch.device):
    if sample_steps >= T:
        return torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)
    t_seq = torch.linspace(T - 1, 0, steps=sample_steps, device=device).long()
    t_seq = torch.unique(t_seq, sorted=True)
    return torch.flip(t_seq, dims=[0])  # descending


def pearson_corr_per_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a,b: [B, N]
    returns: [B] correlation across N per sample
    """
    a_mean = a.mean(dim=1, keepdim=True)
    b_mean = b.mean(dim=1, keepdim=True)
    a0 = a - a_mean
    b0 = b - b_mean
    cov = (a0 * b0).mean(dim=1)
    a_std = a0.pow(2).mean(dim=1).sqrt().clamp_min(eps)
    b_std = b0.pow(2).mean(dim=1).sqrt().clamp_min(eps)
    return cov / (a_std * b_std)


def p2_weight_from_alpha_bar(alpha_bar_t: torch.Tensor, k: float = 1.0, gamma: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    alpha_bar_t: [B]  (각 샘플의 누적 alpha_bar)
    returns: [B] p2 weights
    SNR = alpha_bar / (1 - alpha_bar)
    p2 = (k + SNR)^(-gamma)
    """
    ab = alpha_bar_t.clamp(min=eps, max=1.0 - eps)
    snr = ab / (1.0 - ab)
    w = (k + snr).pow(-gamma)
    return w


def v_from_eps_y0(eps: torch.Tensor, y0: torch.Tensor, ab_t: torch.Tensor) -> torch.Tensor:
    """
    eps, y0: [B, N]
    ab_t: [B]  (alpha_bar[t])
    returns v: [B, N]
    v = sqrt(ab)*eps - sqrt(1-ab)*y0
    """
    sqrt_ab = torch.sqrt(ab_t).unsqueeze(1)           # [B,1]
    sqrt_1mab = torch.sqrt(1.0 - ab_t).unsqueeze(1)   # [B,1]
    return sqrt_ab * eps - sqrt_1mab * y0


def y0_from_v_yt(v: torch.Tensor, y_t: torch.Tensor, ab_t: torch.Tensor) -> torch.Tensor:
    """
    v, y_t: [B, N]
    ab_t: [B]
    y0 = sqrt(ab)*y_t - sqrt(1-ab)*v
    """
    sqrt_ab = torch.sqrt(ab_t).unsqueeze(1)
    sqrt_1mab = torch.sqrt(1.0 - ab_t).unsqueeze(1)
    return sqrt_ab * y_t - sqrt_1mab * v


def eps_from_v_yt(v: torch.Tensor, y_t: torch.Tensor, ab_t: torch.Tensor) -> torch.Tensor:
    """
    v, y_t: [B, N]
    ab_t: [B]
    eps = sqrt(1-ab)*y_t + sqrt(ab)*v
    """
    sqrt_ab = torch.sqrt(ab_t).unsqueeze(1)
    sqrt_1mab = torch.sqrt(1.0 - ab_t).unsqueeze(1)
    return sqrt_1mab * y_t + sqrt_ab * v