import csv
from pathlib import Path
from typing import List

import torch
import pickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import set_seed, list_tickers, load_split_tensors, TimeIndexDataset
from model_regression import RegressionTransformer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# =========================
# Global Config
# =========================
BASE_DATA_ROOT = Path("backtest/tensor_data")
BASE_OUT_ROOT  = Path("backtest/regression_runs")
BASE_OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = list(range(3))

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 2048

EXP_LOSS_WEIGHT = 1.0
MAX_EPOCHS_RATE = 50 * 800
MIN_EPOCHS_RATE = 0.6
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
AMP = (DEVICE == "cuda")


# =========================
# Utils
# =========================
def token_mask_ratio(epoch, max_epoch,
                     start=0.3, end=0.0, end_ratio=0.2):
    real_max_epoch = max_epoch - int(end_ratio * max_epoch)
    if epoch <= real_max_epoch:
        alpha = epoch / real_max_epoch
        return start * (1 - alpha) + end * alpha
    else:
        return end


def apply_token_mask(x, mask_ratio):
    if mask_ratio <= 0:
        return x

    B, N, F = x.shape
    mask = torch.rand(B, N, device=x.device) < mask_ratio
    x = x.clone()
    x[mask] = 0.0
    return x


def weighted_ensemble(preds: List[torch.Tensor], weights: List[float]):
    w = torch.tensor(weights, device=preds[0].device)
    w = torch.clamp(w, min=0.0)
    w = w / (w.sum() + 1e-8)

    stacked = torch.stack(preds)  # [M, 2, B, N]
    y_hat = (stacked * w[:, None, None, None]).sum(0)
    return y_hat[0], y_hat[1]


# =========================
# Eval
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    totals = [0.0] * 3
    n_batches = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mask = (x[:, :, 0] != 0).float()

        y_hat, confi = model(x)

        # evaluate training loss
        masked_confi = confi * mask
        masked_confi = masked_confi / masked_confi.sum(dim=1, keepdim=True)

        loss_bin = (
            F.binary_cross_entropy_with_logits(
                y_hat,
                y.float(),
                reduction='none'
            ) * mask
        ).sum() / (mask.sum() + 1e-8)

        loss = loss_bin * EXP_LOSS_WEIGHT
        totals[0] += loss.item()
        totals[1] += loss_bin.item()

        # evaluate complimental curves
        y_hat *= mask

        _, topk = torch.topk(y_hat, k=5, dim=1)
        clipped = torch.zeros_like(y_hat)
        clipped.scatter_(1, topk, 1.0)
        denom = torch.sum(clipped, dim=1).mean() + 1e-8
        exp  = (torch.sum(y * clipped, dim=1) / denom).mean()
        totals[2] += exp.item()

        n_batches += 1

    return [t / n_batches for t in totals]


@torch.no_grad()
def evaluate_ensemble(models, weights, loader):
    for m in models:
        m.eval()

    totals = [0.0] * 3
    n_batches = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mask = (x[:, :, 0] != 0).float()

        preds = [torch.stack(m(x)) for m in models]
        y_hat, confi = weighted_ensemble(preds, weights)

        # evaluate training loss
        masked_confi = confi * mask
        masked_confi = masked_confi / (masked_confi.sum(dim=1, keepdim=True) + 1e-8)

        loss_bin = (
            F.binary_cross_entropy_with_logits(
                y_hat,
                y.float(),
                reduction='none'
            ) * mask
        ).sum() / (mask.sum() + 1e-8)

        loss = loss_bin * EXP_LOSS_WEIGHT
        
        totals[0] += loss.item()
        totals[1] += loss_bin.item()

        # evaluate complimental curves
        y_hat *= mask

        _, topk = torch.topk(y_hat, k=5, dim=1)
        clipped = torch.zeros_like(y_hat)
        clipped.scatter_(1, topk, 1.0)
        denom = torch.sum(clipped, dim=1).mean() + 1e-8
        exp  = (torch.sum(y * clipped, dim=1) / denom).mean()
        totals[2] += exp.item()

        n_batches += 1

    return [t / n_batches for t in totals]


# =========================
# Train
# =========================
def train_one_epoch(model, loader, optimizer, scaler, scheduler, epoch, epoch_max):
    model.train()
    mask_ratio = token_mask_ratio(epoch, epoch_max)

    running = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        predictable = (x[:, :, 0] != 0).float()

        # mask = predictable
        # valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        # mean = (y * mask).sum(dim=1, keepdim=True) / valid_count
        # var = ((y - mean) * mask).pow(2).sum(dim=1, keepdim=True) / valid_count
        # std = var.sqrt().clamp(min=1e-6)
        # y_norm = (y - mean) / std

        x = apply_token_mask(x, mask_ratio)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=AMP):
            y_hat, confi = model(x)
            masked_confi = confi * predictable
            masked_confi = masked_confi / masked_confi.sum(dim=1, keepdim=True)

            # loss_mse = ((y_hat - y_norm) ** 2 * predictable).sum() / (predictable.sum() + 1e-8)
            # loss_exp = torch.sum(y * masked_confi, dim=1).mean()
            loss_bin = (
                F.binary_cross_entropy_with_logits(
                    y_hat,
                    y.float(),
                    reduction='none'
                ) * predictable
            ).sum() / (predictable.sum() + 1e-8)

            loss = loss_bin * EXP_LOSS_WEIGHT

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        n_batches += 1

    scheduler.step()
    return running / n_batches


# =========================
# Main Backtest Loop
# =========================
def main():
    date_dirs = sorted([d for d in BASE_DATA_ROOT.iterdir() if d.is_dir()])

    test_val_lst = []

    for date_dir in date_dirs:
        date = date_dir.name

        DATA_ROOT = date_dir
        OUT_DIR = BASE_OUT_ROOT / date
        LOG_CSV = OUT_DIR / "metrics.csv"

        if LOG_CSV.exists():
            print(f"[SKIP] {date} already trained")
            continue

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        train_dir = DATA_ROOT / "train"
        val_dir   = DATA_ROOT / "val"
        test_dir  = DATA_ROOT / "test"

        tickers = list_tickers(train_dir)
        X_train, Y_train = load_split_tensors(train_dir, tickers)
        X_val,   Y_val   = load_split_tensors(val_dir, tickers)
        X_test,  Y_test  = load_split_tensors(test_dir, tickers)

        train_loader = DataLoader(TimeIndexDataset(X_train, Y_train),
                                  batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader   = DataLoader(TimeIndexDataset(X_val, Y_val),
                                  batch_size=TEST_BATCH_SIZE)
        test_loader  = DataLoader(TimeIndexDataset(X_test, Y_test),
                                  batch_size=TEST_BATCH_SIZE)

        csv_rows = []
        best_models = []
        best_scores = []
        epoch_max = int(MAX_EPOCHS_RATE / len(X_train))
        epoch_warmup = int(epoch_max / 15)

        for seed in SEEDS:
            set_seed(seed)

            model = RegressionTransformer(
                n_tokens=X_train.shape[1],
                in_dim=X_train.shape[2],
                d_model=128,
                n_head=8,
                n_layers=3,
                d_ff=256,
                dropout=0.1,
            ).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scaler = torch.amp.GradScaler("cuda", enabled=AMP)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=epoch_warmup
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=epoch_max - epoch_warmup,
                eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[epoch_warmup]
            )

            best_score = 1e9
            ckpt = OUT_DIR / f"best_model_seed{seed}.pt"

            for epoch in range(1, epoch_max + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, epoch, epoch_max)
                vals = evaluate(model, val_loader)

                if epoch > 1:
                    csv_rows.append([seed, epoch, train_loss, *vals, optimizer.param_groups[0]["lr"]])

                if vals[0] < best_score and epoch > epoch_max * MIN_EPOCHS_RATE:
                    best_score = vals[0]
                    torch.save(model.state_dict(), ckpt)

            model.load_state_dict(torch.load(ckpt))
            best_models.append(model)
            best_scores.append(1.5-best_score)

        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed", "epoch", "train_loss",
                "val_loss", "val_bin_loss", "val_confi_exp", "lr"
            ])
            writer.writerows(csv_rows)

        # =========================
        # Plot
        # =========================
        import pandas as pd
        df = pd.read_csv(LOG_CSV)

        for col in ["train_loss", "val_loss", "val_bin_loss", "val_confi_exp", "lr"]:
            plt.figure()
            for seed in SEEDS:
                d = df[df.seed == seed]
                plt.plot(d.epoch, d[col], label=f"seed={seed}")
            plt.legend()
            plt.title(col)
            plt.savefig(OUT_DIR / f"{col}.png")
            plt.close()

        test_vals = evaluate_ensemble(best_models, best_scores, test_loader)
        print(f"[{date} Ensemble Test] | bin_loss: {test_vals[1]}, exp5: {test_vals[2]}")
        test_val_lst.append(test_vals)

        # 🔥 GPU 메모리 정리
        for m in best_models:
            del m
        best_models.clear()
        best_scores.clear()

        torch.cuda.empty_cache()

    test_val_lst_path = BASE_OUT_ROOT / "test_val_lst.pkl"
    with open(test_val_lst_path, "wb") as f:
        pickle.dump(test_val_lst, f)


if __name__ == "__main__":
    main()