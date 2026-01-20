# train_regression.py
import csv
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import set_seed, list_tickers, load_split_tensors, TimeIndexDataset
from model_regression import RegressionTransformer


# =========================
# Config
# =========================
DATA_ROOT = Path("tensor_data")
OUT_DIR = Path("regression_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 2048

EPOCHS = 30
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
AMP = (DEVICE == "cuda")

LOG_CSV = OUT_DIR / "metrics.csv"
BEST_CKPT = OUT_DIR / "best_model.pt"


# =========================
# Eval
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_rmse = total_mae = total_corr = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        mask = (x[:, :, 0] != 0).float()
        y_hat = model(x)

        diff = (y_hat - y) * mask
        denom = mask.sum() + 1e-8

        rmse = torch.sqrt(diff.pow(2).sum() / denom)
        mae = diff.abs().sum() / denom

        # corr
        m = mask
        vc = m.sum(dim=1).clamp_min(1.0)

        a_mean = (y_hat * m).sum(dim=1, keepdim=True) / vc.unsqueeze(1)
        b_mean = (y * m).sum(dim=1, keepdim=True) / vc.unsqueeze(1)

        a0 = (y_hat - a_mean) * m
        b0 = (y - b_mean) * m

        cov = (a0 * b0).sum(dim=1) / vc
        a_std = (a0.pow(2).sum(dim=1) / vc).sqrt().clamp_min(1e-8)
        b_std = (b0.pow(2).sum(dim=1) / vc).sqrt().clamp_min(1e-8)
        corr = (cov / (a_std * b_std)).mean()

        total_rmse += rmse.item()
        total_mae += mae.item()
        total_corr += corr.item()
        n_batches += 1

    return (
        total_rmse / n_batches,
        total_mae / n_batches,
        total_corr / n_batches,
    )


# =========================
# Train
# =========================
def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    running = 0.0
    n_batches = 0

    for x, y in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        mask = (x[:, :, 0] != 0).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=AMP):
            y_hat = model(x)
            loss = ((y_hat - y).pow(2) * mask).sum() / (mask.sum() + 1e-8)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{running / n_batches:.6f}")

    return running / n_batches


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    test_dir = DATA_ROOT / "test"

    tickers = list_tickers(train_dir)

    X_train, Y_train = load_split_tensors(train_dir, tickers)
    X_val, Y_val = load_split_tensors(val_dir, tickers)
    X_test, Y_test = load_split_tensors(test_dir, tickers)

    T, N, F = X_train.shape

    train_loader = DataLoader(
        TimeIndexDataset(X_train, Y_train),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TimeIndexDataset(X_val, Y_val),
        batch_size=TEST_BATCH_SIZE,
    )
    test_loader = DataLoader(
        TimeIndexDataset(X_test, Y_test),
        batch_size=TEST_BATCH_SIZE,
    )

    model = RegressionTransformer(
        n_tokens=N,
        in_dim=F,
        d_model=96,
        n_head=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    best_rmse = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        val_rmse, val_mae, val_corr = evaluate(model, val_loader)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | "
            f"val_rmse={val_rmse:.6f} "
            f"val_mae={val_mae:.6f} "
            f"val_corr={val_corr:.4f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), BEST_CKPT)

    model.load_state_dict(torch.load(BEST_CKPT))
    test_rmse, test_mae, test_corr = evaluate(model, test_loader)

    print(f"[Test] rmse={test_rmse:.6f} mae={test_mae:.6f} corr={test_corr:.4f}")


if __name__ == "__main__":
    main()