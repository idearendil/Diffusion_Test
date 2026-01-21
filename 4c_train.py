# train_regression_ensemble_fixed.py
import csv
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from utils import set_seed, list_tickers, load_split_tensors, TimeIndexDataset
from model_regression import RegressionTransformer


# =========================
# Config
# =========================
DATA_ROOT = Path("tensor_data")
OUT_DIR = Path("regression_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = (DEVICE == "cuda")

SEEDS = [42, 43, 44]

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 2048

EPOCHS = 40
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

LOG_CSV = OUT_DIR / "metrics.csv"


# =========================
# Token Masking Schedule
# =========================
def token_mask_ratio(epoch, max_epoch, start=0.3, end=0.0):
    alpha = epoch / max_epoch
    return start * (1 - alpha) + end * alpha


def apply_token_mask(x, mask_ratio):
    if mask_ratio <= 0:
        return x

    B, N, F = x.shape
    mask = torch.rand(B, N, device=x.device) < mask_ratio
    x = x.clone()
    x[mask] = 0.0
    return x


# =========================
# Loss
# =========================
def corr_loss(y_hat, y, mask):
    vc = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    y_hat0 = (y_hat - (y_hat * mask).sum(1, keepdim=True) / vc) * mask
    y0 = (y - (y * mask).sum(1, keepdim=True) / vc) * mask

    cov = (y_hat0 * y0).sum(1)
    std = torch.sqrt((y_hat0**2).sum(1) * (y0**2).sum(1) + 1e-8)

    corr = cov / std
    return 1 - corr.mean()


# =========================
# Weighted Ensemble
# =========================
def weighted_ensemble(preds: List[torch.Tensor], weights: List[float]):
    w = torch.tensor(weights, device=preds[0].device)
    w = torch.clamp(w, min=0.0)
    w = w / (w.sum() + 1e-8)

    stacked = torch.stack(preds)  # [M, B, N]
    return (stacked * w[:, None, None]).sum(0)


# =========================
# Evaluation (single model)
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_rmse = total_mae = total_corr = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # z-score per sample
        y = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-6)

        mask = (x.abs().sum(dim=2) != 0).float()
        y_hat = model(x)

        diff = (y_hat - y) * mask
        denom = mask.sum() + 1e-8

        rmse = torch.sqrt(diff.pow(2).sum() / denom)
        mae = diff.abs().sum() / denom

        vc = mask.sum(dim=1).clamp_min(1.0)
        a0 = (y_hat - (y_hat * mask).sum(1, keepdim=True) / vc.unsqueeze(1)) * mask
        b0 = (y - (y * mask).sum(1, keepdim=True) / vc.unsqueeze(1)) * mask

        corr = ((a0 * b0).sum(1) /
                (torch.sqrt((a0**2).sum(1) * (b0**2).sum(1)) + 1e-8)).mean()

        total_rmse += rmse.item()
        total_mae += mae.item()
        total_corr += corr.item()
        n_batches += 1

    return total_rmse / n_batches, total_mae / n_batches, total_corr / n_batches


# =========================
# Evaluation (ensemble)
# =========================
@torch.no_grad()
def evaluate_ensemble(models, weights, loader):
    for m in models:
        m.eval()

    total_rmse = total_mae = total_corr = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-6)
        mask = (x.abs().sum(dim=2) != 0).float()

        preds = [m(x) for m in models]
        y_hat = weighted_ensemble(preds, weights)

        diff = (y_hat - y) * mask
        denom = mask.sum() + 1e-8

        rmse = torch.sqrt(diff.pow(2).sum() / denom)
        mae = diff.abs().sum() / denom

        vc = mask.sum(dim=1).clamp_min(1.0)
        a0 = (y_hat - (y_hat * mask).sum(1, keepdim=True) / vc.unsqueeze(1)) * mask
        b0 = (y - (y * mask).sum(1, keepdim=True) / vc.unsqueeze(1)) * mask

        corr = ((a0 * b0).sum(1) /
                (torch.sqrt((a0**2).sum(1) * (b0**2).sum(1)) + 1e-8)).mean()

        total_rmse += rmse.item()
        total_mae += mae.item()
        total_corr += corr.item()
        n_batches += 1

    return total_rmse / n_batches, total_mae / n_batches, total_corr / n_batches


# =========================
# Train
# =========================
def train_one_epoch(model, loader, optimizer, scaler, scheduler, epoch):
    model.train()
    mask_ratio = token_mask_ratio(epoch, EPOCHS)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, ncols=120)
    running = 0.0
    n_batches = 0

    for x, y in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-6)

        # 🔑 mask는 token masking 이전 기준
        mask = (x.abs().sum(dim=2) != 0).float()
        x = apply_token_mask(x, mask_ratio)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=AMP):
            y_hat = model(x)
            loss_mse = ((y_hat - y).pow(2) * mask).sum() / (mask.sum() + 1e-8)
            loss_corr = corr_loss(y_hat, y, mask)
            loss = 0.5 * loss_mse + 0.5 * loss_corr

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{running / n_batches:.5f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "mask": f"{mask_ratio:.2f}",
        })

    return running / n_batches


# =========================
# Main
# =========================
def main():
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    test_dir = DATA_ROOT / "test"

    tickers = list_tickers(train_dir)

    X_train, Y_train = load_split_tensors(train_dir, tickers)
    X_val, Y_val = load_split_tensors(val_dir, tickers)
    X_test, Y_test = load_split_tensors(test_dir, tickers)

    train_loader = DataLoader(
        TimeIndexDataset(X_train, Y_train),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TimeIndexDataset(X_val, Y_val),
        batch_size=TEST_BATCH_SIZE
    )
    test_loader = DataLoader(
        TimeIndexDataset(X_test, Y_test),
        batch_size=TEST_BATCH_SIZE
    )

    csv_rows = []
    best_models = []
    best_corrs = []

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n===== SEED {seed} =====")

        model = RegressionTransformer(
            n_tokens=X_train.shape[1],
            in_dim=X_train.shape[2],
            d_model=128,
            n_head=8,
            n_layers=3,
            d_ff=256,
            dropout=0.1,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        total_steps = EPOCHS * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=LR * 0.05
        )

        scaler = torch.amp.GradScaler("cuda", enabled=AMP)

        best_corr = -1.0
        ckpt_path = OUT_DIR / f"best_model_seed{seed}.pt"

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, scheduler, epoch
            )
            val_rmse, val_mae, val_corr = evaluate(model, val_loader)

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.5f} "
                f"val_rmse={val_rmse:.5f} "
                f"val_mae={val_mae:.5f} "
                f"val_corr={val_corr:.5f}"
            )

            csv_rows.append([
                seed, epoch, train_loss, val_rmse, val_mae, val_corr
            ])

            if val_corr > best_corr:
                best_corr = val_corr
                torch.save(model.state_dict(), ckpt_path)

        model.load_state_dict(torch.load(ckpt_path))
        best_models.append(model)
        best_corrs.append(best_corr)

    # =========================
    # Save CSV
    # =========================
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "epoch", "train_loss", "val_rmse", "val_mae", "val_corr"])
        writer.writerows(csv_rows)

    # =========================
    # Ensemble Test
    # =========================
    print("\nEnsemble weights (val_corr):", best_corrs)
    test_rmse, test_mae, test_corr = evaluate_ensemble(best_models, best_corrs, test_loader)

    print(
        f"[Weighted Ensemble Test] "
        f"rmse={test_rmse:.6f} "
        f"mae={test_mae:.6f} "
        f"corr={test_corr:.4f}"
    )


if __name__ == "__main__":
    main()