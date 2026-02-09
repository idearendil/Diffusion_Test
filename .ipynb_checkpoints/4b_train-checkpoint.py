# train.py
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    set_seed, list_tickers, load_split_tensors, TimeIndexDataset,
    cosine_beta_schedule, make_t_seq,
    p2_weight_from_alpha_bar,
    v_from_eps_y0, y0_from_v_yt, eps_from_v_yt
)
from model import DiffusionTransformer


# =========================
# Config
# =========================
# ✅ backtest 루트들
BACKTEST_DATA_ROOT = Path("backtest/tensor_data")   # backtest/tensor_data/<run>/train,val,test
BACKTEST_OUT_ROOT  = Path("backtest/models")        # backtest/models/<run>/
BACKTEST_OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Data
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 2048
NUM_WORKERS = 0
PIN_MEMORY = (DEVICE == "cuda")

# Diffusion
T_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

SAMPLE_STEPS = 100
DDIM_ETA = 0.0

# Model
D_MODEL = 96
N_HEAD = 4
N_LAYERS = 3
D_FF = 256
DROPOUT = 0.1

# p2 loss weighting
P2_GAMMA = 1.0
P2_K = 1.0

# Training
EPOCHS = 30
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
AMP = True if DEVICE == "cuda" else False


# =========================
# Sampling
# =========================
@torch.no_grad()
def ddim_sample_y0(
    model: torch.nn.Module,
    x: torch.Tensor,                 # [B, N, F]
    diffusion: Dict[str, torch.Tensor],
    t_seq: torch.Tensor,
    eta: float = 0.0,
) -> torch.Tensor:
    model.eval()
    alpha_bar = diffusion["alpha_bar"]  # [T]

    B, N, _ = x.shape
    y = torch.randn(B, N, device=x.device)

    for i, t_i in enumerate(t_seq):
        t = torch.full((B,), int(t_i.item()), device=x.device, dtype=torch.long)

        tokens = torch.cat([y.unsqueeze(-1), x], dim=-1)  # [B,N,F+1]
        v_pred = model(tokens, t)                         # [B,N]

        ab_t_scalar = alpha_bar[t_i]
        ab_t = torch.full((B,), float(ab_t_scalar.item()), device=x.device, dtype=torch.float32)

        y0_hat = y0_from_v_yt(v_pred, y, ab_t)        # [B,N]
        eps_pred = eps_from_v_yt(v_pred, y, ab_t)     # [B,N]

        if i == len(t_seq) - 1:
            y = y0_hat
            break

        t_prev = t_seq[i + 1]
        ab_prev_scalar = alpha_bar[t_prev]
        sqrt_ab_prev = torch.sqrt(ab_prev_scalar)

        if eta > 0.0:
            sigma = eta * torch.sqrt((1 - ab_prev_scalar) / (1 - ab_t_scalar)) * torch.sqrt(1 - ab_t_scalar / ab_prev_scalar)
            z = torch.randn_like(y)
        else:
            sigma = 0.0
            z = 0.0

        c = torch.sqrt(torch.clamp(1.0 - ab_prev_scalar - sigma**2, min=0.0))
        y = sqrt_ab_prev * y0_hat + c * eps_pred + sigma * z

    return y  # [B,N]


@torch.no_grad()
def ddim_sample_y0_kmean_var(
    model: torch.nn.Module,
    x: torch.Tensor,                 # [B, N, F]
    diffusion: Dict[str, torch.Tensor],
    t_seq: torch.Tensor,
    k: int = 10,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    samples = []
    for _ in range(k):
        y0_hat = ddim_sample_y0(model, x, diffusion, t_seq=t_seq, eta=eta)
        samples.append(y0_hat)

    S = torch.stack(samples, dim=0)          # [K,B,N]
    y0_mean = S.mean(dim=0)                  # [B,N]
    y0_var  = S.var(dim=0, unbiased=False)   # [B,N]
    return y0_mean, y0_var


# =========================
# Eval / Train
# =========================
@torch.no_grad()
def evaluate_sampling(
    model: torch.nn.Module,
    loader: DataLoader,
    diffusion: Dict[str, torch.Tensor],
    sample_steps: int,
    k_samples: int = 1,
    eta: float = 0.0,
) -> Tuple[float, float, float, float]:
    model.eval()

    total_rmse = 0.0
    total_mae  = 0.0
    total_corr = 0.0
    total_var  = 0.0
    n_batches = 0

    for x, y0 in loader:
        x = x.to(DEVICE)
        y0 = y0.to(DEVICE)

        mask = (x[:, :, 0] != 0).float()

        t_seq = make_t_seq(T_STEPS, sample_steps, x.device)
        y0_hat, y0_var = ddim_sample_y0_kmean_var(
            model, x, diffusion, t_seq=t_seq, k=k_samples, eta=eta
        )

        diff = (y0_hat - y0) * mask
        denom = mask.sum() + 1e-8
        rmse = torch.sqrt(diff.pow(2).sum() / denom)
        mae  = diff.abs().sum() / denom

        # corr (mask 반영)
        eps = 1e-8
        m = mask
        valid_counts = m.sum(dim=1).clamp_min(1.0)

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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    diffusion: Dict[str, torch.Tensor],
    epoch: int,
) -> float:
    model.train()

    sqrt_ab = diffusion["sqrt_alpha_bar"]
    sqrt_1mab = diffusion["sqrt_one_minus_alpha_bar"]
    alpha_bar = diffusion["alpha_bar"]

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    running = 0.0
    n_batches = 0

    for x, y0 in pbar:
        x = x.to(DEVICE)
        y0 = y0.to(DEVICE)
        B = x.shape[0]

        t = torch.randint(0, T_STEPS, (B,), device=DEVICE, dtype=torch.int64)
        eps = torch.randn_like(y0)
        y_t = sqrt_ab[t].unsqueeze(1) * y0 + sqrt_1mab[t].unsqueeze(1) * eps
        tokens = torch.cat([y_t.unsqueeze(-1), x], dim=-1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=AMP):
            v_pred = model(tokens, t)

            mask = (x[:, :, 0] != 0).float()
            ab_t = alpha_bar[t]

            v_tgt = v_from_eps_y0(eps, y0, ab_t)
            token_mse = (v_pred - v_tgt).pow(2)

            w = p2_weight_from_alpha_bar(ab_t, k=P2_K, gamma=P2_GAMMA).unsqueeze(1)
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


# =========================
# Backtest helpers
# =========================
def find_backtest_runs(root: Path) -> List[Path]:
    """
    backtest/tensor_data 아래에서
    - 하위 폴더 중 train/val/test 디렉토리가 모두 존재하는 폴더를 '런'으로 간주
    """
    runs: List[Path] = []
    if not root.exists():
        raise FileNotFoundError(f"Backtest data root not found: {root}")

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "train").is_dir() and (d / "val").is_dir() and (d / "test").is_dir():
            runs.append(d)

    return runs


def train_one_backtest_run(data_root: Path, out_dir: Path):
    """
    기존 main() 내용을 '단일 데이터셋' 학습으로 캡슐화.
    out_dir 안에 metrics.csv, curves.png, best_model.pt 저장.
    """
    set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_csv = out_dir / "metrics.csv"
    plot_png = out_dir / "curves.png"
    best_ckpt = out_dir / "best_model.pt"

    print("=" * 80)
    print(f"[RUN] data_root={data_root}  -> out_dir={out_dir}")
    print("Device:", DEVICE)

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    tickers = list_tickers(train_dir)
    if not tickers:
        raise FileNotFoundError(f"No *_x.pt files in {train_dir}")
    print(f"Tickers: {len(tickers)}")

    for split_name, split_dir in [("val", val_dir), ("test", test_dir)]:
        split_tickers = set(list_tickers(split_dir))
        missing = [t for t in tickers if t not in split_tickers]
        if missing:
            raise ValueError(f"Missing tickers in {split_name}: {missing[:10]} ... total {len(missing)}")

    X_train, Y_train = load_split_tensors(train_dir, tickers)
    X_val, Y_val = load_split_tensors(val_dir, tickers)
    X_test, Y_test = load_split_tensors(test_dir, tickers)

    T_train, N, Fdim = X_train.shape
    print(f"Train X: {tuple(X_train.shape)}  Train Y: {tuple(Y_train.shape)}")
    print(f"Val   X: {tuple(X_val.shape)}  Val   Y: {tuple(Y_val.shape)}")
    print(f"Test  X: {tuple(X_test.shape)} Test  Y: {tuple(Y_test.shape)}")

    train_loader = DataLoader(
        TimeIndexDataset(X_train, Y_train),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        TimeIndexDataset(X_val, Y_val),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    test_loader = DataLoader(
        TimeIndexDataset(X_test, Y_test),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    betas = cosine_beta_schedule(T_STEPS, s=0.008, max_beta=BETA_END).to(DEVICE)
    alphas64 = (1.0 - betas).to(torch.float64)
    alpha_bar = torch.cumprod(alphas64, dim=0).clamp(1e-12, 1.0).to(torch.float32)
    alphas = (1.0 - betas)

    diffusion = {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }

    in_dim = Fdim + 1
    model = DiffusionTransformer(
        n_tokens=N, in_dim=in_dim, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    log_rows: List[Dict[str, float]] = []
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_rmse", "val_mae", "val_corr", "val_var"])
        writer.writeheader()

    best_val_rmse = float("inf")
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, diffusion, epoch)

        val_rmse, val_mae, val_corr, val_var = evaluate_sampling(
            model, val_loader, diffusion, sample_steps=SAMPLE_STEPS, k_samples=1, eta=DDIM_ETA
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

        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch

            ckpt = {
                "model_state": model.state_dict(),
                "tickers": tickers,
                "in_dim": in_dim,
                "Fdim": Fdim,
                "N": N,
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse,
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
            torch.save(ckpt, best_ckpt)
            print(f"✅ Saved best checkpoint (val_rmse={best_val_rmse:.6f}) at epoch {best_epoch}: {best_ckpt}")

    # best 로드 후 test 평가
    if best_ckpt.exists():
        best = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(best["model_state"])
        print(f"Loaded best checkpoint: epoch={best.get('best_epoch')} val_rmse={best.get('best_val_rmse'):.6f}")

    test_rmse, test_mae, test_corr, test_var = evaluate_sampling(
        model, test_loader, diffusion, sample_steps=SAMPLE_STEPS, k_samples=1, eta=DDIM_ETA
    )
    print(f"[Test(best)] rmse={test_rmse:.6f} mae={test_mae:.6f} corr={test_corr:.4f} var={test_var:.4f}")

    plot_curves(log_rows, plot_png)
    print(f"Saved metrics: {log_csv}")
    print(f"Saved plot:    {plot_png}")
    print(f"Best ckpt:     {best_ckpt} (epoch={best_epoch}, val_rmse={best_val_rmse:.6f})")


def main():
    runs = find_backtest_runs(BACKTEST_DATA_ROOT)
    if not runs:
        raise FileNotFoundError(
            f"No backtest runs found under {BACKTEST_DATA_ROOT}. "
            f"Expected: {BACKTEST_DATA_ROOT}/<run>/train,val,test"
        )

    print(f"Found backtest runs: {len(runs)}")
    for run_dir in runs:
        run_name = run_dir.name  # 예: 2017-03-01
        out_dir = BACKTEST_OUT_ROOT / run_name
        if out_dir.exists():
            print(f"Skipping existing run: {run_name}")
            continue
        train_one_backtest_run(run_dir, out_dir)


if __name__ == "__main__":
    main()