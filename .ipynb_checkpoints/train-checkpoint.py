import os
import math
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# pip install torchsummary  (필요하면)
from torchsummary import summary

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
DATA_ROOT = Path("tensor_data")  # tensor_data/train, val, test
OUT_DIR = Path("diffusion_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Data
BATCH_SIZE = 8          # N(토큰 수)~900이면 O(N^2)라 batch 크게 못 잡는 경우 많음
NUM_WORKERS = 0         # pt 파일 메모리로 올리면 CPU 워커 필요 적음
PIN_MEMORY = (DEVICE == "cuda")

# Diffusion
T_STEPS = 1000           # 100~1000 중 선택, 여기선 학습 안정/속도 절충
BETA_START = 1e-4
BETA_END = 0.02

SAMPLE_STEPS = 100   # 50~200 추천. T_STEPS=1000이면 100 정도가 현실적
DDIM_ETA = 0.0       # 0이면 deterministic (DDIM). >0이면 stochastic

# Model (토큰 길이 N이 크므로 작게)
D_MODEL = 96
N_HEAD = 4
N_LAYERS = 3
D_FF = 256
DROPOUT = 0.1

# p2 loss weighting (SNR 기반)
P2_GAMMA = 1.0     # 보통 0.5~1.0 많이 씀
P2_K = 1.0         # 보통 1.0

# Training
EPOCHS = 300
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
AMP = True if DEVICE == "cuda" else False  # mixed precision

LOG_CSV = OUT_DIR / "metrics.csv"
PLOT_PNG = OUT_DIR / "curves.png"


# =========================
# Utils
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def make_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


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

# =========================
# Embeddings
# =========================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] int64
        returns: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device, dtype=torch.float32) * (-1)
        )
        # [B, half]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device)], dim=1)
        return emb


# =========================
# Model
# =========================
class DiffusionTransformer(nn.Module):
    """
    Input tokens: [B, N, (F+1)]  where first dim is y_t (noisy target), rest is conditioning x
    Output: eps_pred [B, N] (only first scalar per token is used conceptually)
    """
    def __init__(self, n_tokens: int, in_dim: int, d_model: int, n_head: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_tokens = n_tokens
        self.in_dim = in_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(in_dim, d_model)

        # Learned positional encoding (token position = ticker position)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        # Time embedding -> project to d_model and add to all tokens
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, in_dim]
        t: [B] int64
        returns: eps_pred [B, N]
        """
        h = self.in_proj(tokens)  # [B,N,d]
        h = h + self.pos_emb  # positional encoding

        te = self.time_mlp(self.time_emb(t))  # [B,d]
        h = h + te.unsqueeze(1)  # add to all tokens

        h = self.encoder(h)  # [B,N,d]
        h = self.out_norm(h)
        eps = self.pred_head(h).squeeze(-1)  # [B,N]
        return eps


# =========================
# Train / Eval
# =========================
@torch.no_grad()
def evaluate_sampling(
    model: nn.Module,
    loader: DataLoader,
    diffusion: Dict[str, torch.Tensor],
    sample_steps: int,
    k_samples: int = 10,
    eta: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    K번 샘플링 평균으로 RMSE/MAE/Corr 계산 + 샘플 분산(val_var)도 로그
    val_var: 토큰별 샘플분산을 구한 뒤, 마스크된 토큰만 평균낸 값
    """
    model.eval()

    total_rmse = 0.0
    total_mae  = 0.0
    total_corr = 0.0
    total_var  = 0.0
    n_batches = 0

    for x, y0 in loader:
        x = x.to(DEVICE)      # [B,N,F]
        y0 = y0.to(DEVICE)    # [B,N]

        mask = (x[:, :, 0] != 0).float()  # [B,N]

        # ===== K번 샘플링 평균 + 분산 =====
        y0_hat, y0_var = ddim_sample_y0_kmean_var(
            model, x, diffusion,
            sample_steps=sample_steps,
            k=k_samples,
            eta=eta
        )  # [B,N], [B,N]

        # ===== RMSE/MAE =====
        diff = (y0_hat - y0) * mask
        denom = mask.sum() + 1e-8
        rmse = torch.sqrt(diff.pow(2).sum() / denom)
        mae  = diff.abs().sum() / denom

        # ===== Corr (mask 반영) =====
        eps = 1e-8
        m = mask
        valid_counts = m.sum(dim=1).clamp_min(1.0)  # [B]

        a = y0_hat
        b = y0
        a_mean = (a * m).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(1)
        b_mean = (b * m).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(1)

        a0 = (a - a_mean) * m
        b0 = (b - b_mean) * m

        cov = (a0 * b0).sum(dim=1) / valid_counts
        a_std = (a0.pow(2).sum(dim=1) / valid_counts).sqrt().clamp_min(eps)
        b_std = (b0.pow(2).sum(dim=1) / valid_counts).sqrt().clamp_min(eps)
        corr = (cov / (a_std * b_std)).mean()

        # ===== val_var: 토큰별 분산의 마스크 평균 =====
        # (B,N) -> 스칼라
        var_mean = (y0_var * mask).sum() / (mask.sum() + 1e-8)

        total_rmse += rmse.item()
        total_mae  += mae.item()
        total_corr += corr.item()
        total_var  += var_mean.item()
        n_batches += 1

    return (
        total_rmse / max(n_batches, 1),
        total_mae  / max(n_batches, 1),
        total_corr / max(n_batches, 1),
        total_var  / max(n_batches, 1),
    )


@torch.no_grad()
def ddim_sample_y0(
    model: nn.Module,
    x: torch.Tensor,                 # [B, N, F]
    diffusion: Dict[str, torch.Tensor],
    sample_steps: int,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    조건 x를 고정하고 y_T~N(0,1)에서 시작해 DDIM으로 y0_hat 샘플링.
    returns y0_hat: [B, N]
    """
    model.eval()

    alpha_bar = diffusion["alpha_bar"]                 # [T]
    sqrt_ab = diffusion["sqrt_alpha_bar"]              # [T]
    sqrt_1mab = diffusion["sqrt_one_minus_alpha_bar"]  # [T]

    B, N, Fdim = x.shape
    T = alpha_bar.shape[0]

    # timesteps를 stride로 줄여서 샘플링 (ex: 999, 989, ..., 0)
    # 내림차순 리스트
    if sample_steps >= T:
        t_seq = torch.arange(T - 1, -1, -1, device=x.device, dtype=torch.long)
    else:
        t_seq = torch.linspace(T - 1, 0, steps=sample_steps, device=x.device).long()
        # 중복 제거 + 내림차순 정렬
        t_seq = torch.unique(t_seq, sorted=True)
        t_seq = torch.flip(t_seq, dims=[0])

    # 시작: y_t (가장 큰 t)
    y = torch.randn(B, N, device=x.device)

    for i, t_i in enumerate(t_seq):
        t = torch.full((B,), int(t_i.item()), device=x.device, dtype=torch.long)
    
        # v 예측
        tokens = torch.cat([y.unsqueeze(-1), x], dim=-1)  # [B,N,F+1]
        v_pred = model(tokens, t)                         # [B,N]
    
        # 현재 ab_t (스칼라 -> [B]로 확장)
        ab_t_scalar = alpha_bar[t_i]  # scalar
        ab_t = torch.full((B,), float(ab_t_scalar.item()), device=x.device, dtype=torch.float32)
    
        # y0 추정 (v-pred)
        y0_hat = y0_from_v_yt(v_pred, y, ab_t)            # [B,N]
    
        # eps 복원 (DDIM update에 필요)
        eps_pred = eps_from_v_yt(v_pred, y, ab_t)         # [B,N]
    
        # 다음(이전) timestep의 alpha_bar
        if i == len(t_seq) - 1:
            y = y0_hat
            break
    
        t_prev = t_seq[i + 1]
        ab_prev_scalar = alpha_bar[t_prev]
        sqrt_ab_prev = torch.sqrt(ab_prev_scalar)
    
        # DDIM sigma (eta=0이면 0)
        if eta > 0.0:
            sigma = eta * torch.sqrt((1 - ab_prev_scalar) / (1 - ab_t_scalar)) * torch.sqrt(1 - ab_t_scalar / ab_prev_scalar)
            z = torch.randn_like(y)
        else:
            sigma = 0.0
            z = 0.0
    
        c = torch.sqrt(torch.clamp(1.0 - ab_prev_scalar - sigma**2, min=0.0))
        y = sqrt_ab_prev * y0_hat + c * eps_pred + sigma * z
    return y  # [B,N] at t=0

@torch.no_grad()
def ddim_sample_y0_kmean_var(
    model: nn.Module,
    x: torch.Tensor,                 # [B, N, F]
    diffusion: Dict[str, torch.Tensor],
    sample_steps: int,
    k: int = 10,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    동일 조건 x에 대해 DDIM 샘플링을 k번 수행.
    returns:
      y0_mean: [B, N]
      y0_var:  [B, N]   (샘플 간 분산, unbiased=False)
    """
    model.eval()
    samples = []
    for _ in range(k):
        y0_hat = ddim_sample_y0(model, x, diffusion, sample_steps=sample_steps, eta=eta)  # [B,N]
        samples.append(y0_hat)

    S = torch.stack(samples, dim=0)          # [K,B,N]
    y0_mean = S.mean(dim=0)                  # [B,N]
    y0_var  = S.var(dim=0, unbiased=False)   # [B,N]
    return y0_mean, y0_var

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                    diffusion: Dict[str, torch.Tensor], epoch: int) -> float:
    model.train()

    sqrt_ab = diffusion["sqrt_alpha_bar"]
    sqrt_1mab = diffusion["sqrt_one_minus_alpha_bar"]

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    running = 0.0
    n_batches = 0

    for x, y0 in pbar:
        x = x.to(DEVICE)       # [B,N,F]
        y0 = y0.to(DEVICE)     # [B,N]

        B = x.shape[0]

        t = torch.randint(0, T_STEPS, (B,), device=DEVICE, dtype=torch.int64)
        eps = torch.randn_like(y0)
        y_t = sqrt_ab[t].unsqueeze(1) * y0 + sqrt_1mab[t].unsqueeze(1) * eps

        tokens = torch.cat([y_t.unsqueeze(-1), x], dim=-1)  # [B,N,F+1]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=AMP):
            v_pred = model(tokens, t)  # [B,N]  (이제 v를 예측한다고 해석)
        
            # token mask: x의 첫 feature가 0이면 loss 0
            mask = (x[:, :, 0] != 0).float()  # [B,N]
        
            alpha_bar = diffusion["alpha_bar"]  # [T]
            ab_t = alpha_bar[t]                 # [B]
        
            # v target
            v_tgt = v_from_eps_y0(eps, y0, ab_t)  # [B,N]
        
            # tokenwise mse on v
            token_mse = (v_pred - v_tgt).pow(2)   # [B,N]
        
            # p2 weighting (SNR 기반)
            w = p2_weight_from_alpha_bar(ab_t, k=P2_K, gamma=P2_GAMMA).unsqueeze(1)  # [B,1]
        
            weighted = token_mse * mask * w
            denom = (mask * w).sum() + 1e-8
            loss = weighted.sum() / denom
            
        scaler.scale(loss).backward()

        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{running / n_batches:.6f}")

    return running / max(n_batches, 1)


def plot_curves(log_rows: List[Dict[str, float]], out_path: Path):
    epochs = [r["epoch"] for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    val_rmse = [r["val_rmse"] for r in log_rows]
    val_mae = [r["val_mae"] for r in log_rows]
    val_corr = [r["val_corr"] for r in log_rows]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_rmse, label="val_rmse")
    plt.plot(epochs, val_mae, label="val_mae")
    plt.plot(epochs, val_corr, label="val_corr")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    test_dir = DATA_ROOT / "test"

    # 1) ticker order fixed by train split filenames
    tickers = list_tickers(train_dir)
    if not tickers:
        raise FileNotFoundError(f"No *_x.pt files in {train_dir}")
    print(f"Tickers: {len(tickers)}")

    # Ensure same tickers exist in val/test
    for split_name, split_dir in [("val", val_dir), ("test", test_dir)]:
        split_tickers = set(list_tickers(split_dir))
        missing = [t for t in tickers if t not in split_tickers]
        if missing:
            raise ValueError(f"Missing tickers in {split_name}: {missing[:10]} ... total {len(missing)}")

    # 2) load tensors
    X_train, Y_train = load_split_tensors(train_dir, tickers)
    X_val, Y_val = load_split_tensors(val_dir, tickers)
    X_test, Y_test = load_split_tensors(test_dir, tickers)

    # Sanity: all same N, F
    T_train, N, Fdim = X_train.shape
    assert N == len(tickers)
    assert X_val.shape[1:] == (N, Fdim)
    assert X_test.shape[1:] == (N, Fdim)

    print(f"Train X: {tuple(X_train.shape)}  Train Y: {tuple(Y_train.shape)}")
    print(f"Val   X: {tuple(X_val.shape)}  Val   Y: {tuple(Y_val.shape)}")
    print(f"Test  X: {tuple(X_test.shape)} Test  Y: {tuple(Y_test.shape)}")

    # 3) datasets/loaders (each index = one day)
    train_ds = TimeIndexDataset(X_train, Y_train)
    val_ds = TimeIndexDataset(X_val, Y_val)
    test_ds = TimeIndexDataset(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

    # 4) diffusion buffers
    betas = cosine_beta_schedule(T_STEPS, s=0.008, max_beta=BETA_END).to(DEVICE)
    alphas = (1.0 - betas).to(torch.float64)
    alpha_bar = torch.cumprod(alphas, dim=0).clamp(1e-12, 1.0).to(torch.float32)
    alphas = (1.0 - betas)  # float32로 다시 저장해도 됨
    
    diffusion = {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }

    # 5) model
    in_dim = Fdim + 1  # y_t + x
    model = DiffusionTransformer(
        n_tokens=N,
        in_dim=in_dim,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    # 6) logging
    log_rows: List[Dict[str, float]] = []
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_rmse", "val_mae", "val_corr", "val_var"])
        writer.writeheader()

    # 7) train loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, diffusion, epoch)
        if epoch % 1 == 0:
            val_rmse, val_mae, val_corr, val_var = evaluate_sampling(
                model, val_loader, diffusion,
                sample_steps=SAMPLE_STEPS,
                k_samples=10,
                eta=DDIM_ETA
            )
            
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f} | "
                f"val_rmse={val_rmse:.6f} val_mae={val_mae:.6f} "
                f"val_corr={val_corr:.4f} val_var={val_var:.6f}"
            )
            
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_corr": val_corr,
                "val_var": val_var,
            }
            log_rows.append(row)

            with open(LOG_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

    # 8) final test eval
    test_rmse, test_mae, test_corr = evaluate_sampling(
        model, test_loader, diffusion,
        sample_steps=SAMPLE_STEPS,
        k_samples=10,
        eta=DDIM_ETA
    )
    print(f"[Test] rmse={test_rmse:.6f} mae={test_mae:.6f} corr={test_corr:.4f}")

    # 9) plot
    plot_curves(log_rows, PLOT_PNG)
    print(f"Saved metrics: {LOG_CSV}")
    print(f"Saved plot:    {PLOT_PNG}")

    # 10) save model + ticker order (중요!)
    ckpt = {
        "model_state": model.state_dict(),
        "tickers": tickers,
        "in_dim": in_dim,
        "Fdim": Fdim,
        "N": N,
        "config": {
            "T_STEPS": T_STEPS,
            "BETA_START": BETA_START,
            "BETA_END": BETA_END,
            "D_MODEL": D_MODEL,
            "N_HEAD": N_HEAD,
            "N_LAYERS": N_LAYERS,
            "D_FF": D_FF,
            "DROPOUT": DROPOUT,
        },
    }
    ckpt_path = OUT_DIR / "model.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()