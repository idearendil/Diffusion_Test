# train_regression_ensemble.py
import csv
from pathlib import Path
from typing import List

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
OUT_DIR = Path("final_regression_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [i for i in range(10)]

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 2048

EXP_LOSS_WEIGHT = 1.0
VAR_LOSS_WEIGHT = 1.0
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
AMP = (DEVICE == "cuda")

LOG_CSV = OUT_DIR / "metrics.csv"

def token_mask_ratio(epoch, max_epoch,
                     start=0.6, end=0.0, end_ratio=0.2):
    """
    epoch 초반엔 mask 많이, 후반엔 0으로 수렴
    """
    real_max_epoch = max_epoch - int(end_ratio * max_epoch)
    if epoch <= real_max_epoch:
        alpha = epoch / real_max_epoch
        return start * (1 - alpha) + end * alpha
    else:
        return end

def apply_token_mask(x, mask_ratio):
    """
    x: [B, N, F]
    """
    if mask_ratio <= 0:
        return x

    B, N, F = x.shape
    device = x.device

    mask = torch.rand(B, N, device=device) < mask_ratio
    x = x.clone()
    x[mask] = 0.0
    return x


def weighted_ensemble(preds: List[torch.Tensor], weights: List[float]):
    """
    preds: list of [B, N] tensors
    weights: list of scalars (val_corr)
    """
    w = torch.tensor(weights, device=preds[0].device)
    w = torch.clamp(w, min=0.0)  # 음수 corr 제거
    w = w / (w.sum() + 1e-8)

    stacked = torch.stack(preds)          # [M, 2, B, N]
    y_hat = (stacked * w[:, None, None, None]).sum(0)
    return y_hat[0], y_hat[1]

# =========================
# Loss
# =========================
def corr_loss(y_hat, y, confi, eps=1e-8):
    """
    y_hat, y, mask, confi: [B, N]
    mask: 0/1
    confi: [0, 1]
    """

    # combined weight
    w = confi                       # [B, N]
    wsum = w.sum(dim=1, keepdim=True).clamp_min(eps)  # [B, 1]

    # weighted mean
    y_hat_mean = (y_hat * w).sum(dim=1, keepdim=True) / wsum
    y_mean     = (y     * w).sum(dim=1, keepdim=True) / wsum

    # centered & weighted
    y_hat0 = (y_hat - y_hat_mean) * w
    y0     = (y     - y_mean)     * w

    # weighted correlation
    cov = (y_hat0 * y0).sum(dim=1)
    std = torch.sqrt(
        (y_hat0 ** 2).sum(dim=1) * (y0 ** 2).sum(dim=1) + eps
    )

    corr = cov / std
    return 1.0 - corr.mean()

# =========================
# Eval (single model)
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_rmse = total_mae = total_exp = total_var = total_confi_rmse = total_confi_mae = total_confi_exp = total_confi_var = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # y = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-6)

        mask = (x[:, :, 0] != 0).float()
        y_hat, confi = model(x)
        confi *= mask

        _, topk_idx = torch.topk(confi, k=3, dim=1)
        clipped_confi = torch.zeros_like(confi)
        clipped_confi.scatter_(1, topk_idx, 1.0)

        # confi 없이 mask만 고려해서 mse, mae, exp, var 계산
        diff = (y_hat - y) * mask
        denom = mask.sum() + 1e-8

        rmse = torch.sqrt(diff.pow(2).sum() / denom)
        mae = diff.abs().sum() / denom
        exp = (torch.sum((y * mask), dim=1) / denom).mean()
        var = (torch.sum((y * mask), dim=1) / denom).var()

        total_rmse += rmse.item()
        total_mae += mae.item()
        total_exp += exp.item()
        total_var += var.item()

        # confi 적용 후 mse, mae, exp, var 계산
        diff = (y_hat - y) * clipped_confi
        denom1 = clipped_confi.sum() + 1e-8
        denom2 = (torch.sum(clipped_confi, dim=1)).mean() + 1e-8

        rmse = torch.sqrt(diff.pow(2).sum() / denom1)
        mae = diff.abs().sum() / denom1
        exp = (torch.sum((y * clipped_confi), dim=1) / denom2).mean()
        var = (torch.sum((y * clipped_confi), dim=1) / denom2).var()

        total_confi_rmse += rmse.item()
        total_confi_mae += mae.item()
        total_confi_exp += exp.item()
        total_confi_var += var.item()

        n_batches += 1

    return (
        total_rmse / n_batches,
        total_mae / n_batches,
        total_exp / n_batches,
        total_var / n_batches,
        total_confi_rmse / n_batches,
        total_confi_mae / n_batches,
        total_confi_exp / n_batches,
        total_confi_var / n_batches,
    )


# =========================
# Eval (ensemble)
# =========================
@torch.no_grad()
def evaluate_ensemble(models, weights, loader):
    for m in models:
        m.eval()

    total_rmse = total_mae = total_exp = total_var = total_confi_rmse = total_confi_mae = total_confi_exp = total_confi_var = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # y = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-6)

        mask = (x[:, :, 0] != 0).float()

        preds = [torch.stack(m(x)) for m in models]
        y_hat, confi = weighted_ensemble(preds, weights)
        confi *= mask

        _, topk_idx = torch.topk(confi, k=3, dim=1)
        clipped_confi = torch.zeros_like(confi)
        clipped_confi.scatter_(1, topk_idx, 1.0)

        # confi 없이 mask만 고려해서 mse, mae, exp, var 계산
        diff = (y_hat - y) * mask
        denom = mask.sum() + 1e-8

        rmse = torch.sqrt(diff.pow(2).sum() / denom)
        mae = diff.abs().sum() / denom
        exp = (torch.sum((y * mask), dim=1) / denom).mean()
        var = (torch.sum((y * mask), dim=1) / denom).var()

        total_rmse += rmse.item()
        total_mae += mae.item()
        total_exp += exp.item()
        total_var += var.item()

        # confi 적용 후 mse, mae, exp, var 계산
        diff = (y_hat - y) * clipped_confi
        denom1 = clipped_confi.sum() + 1e-8
        denom2 = (torch.sum(clipped_confi, dim=1)).mean() + 1e-8

        rmse = torch.sqrt(diff.pow(2).sum() / denom1)
        mae = diff.abs().sum() / denom1
        exp = (torch.sum((y * clipped_confi), dim=1) / denom2).mean()
        var = (torch.sum((y * clipped_confi), dim=1) / denom2).var()

        total_confi_rmse += rmse.item()
        total_confi_mae += mae.item()
        total_confi_exp += exp.item()
        total_confi_var += var.item()

        n_batches += 1

    return (
        total_rmse / n_batches,
        total_mae / n_batches,
        total_exp / n_batches,
        total_var / n_batches,
        total_confi_rmse / n_batches,
        total_confi_mae / n_batches,
        total_confi_exp / n_batches,
        total_confi_var / n_batches,
    )


# =========================
# Train
# =========================
def train_one_epoch(model, loader, optimizer, scaler, scheduler, epoch):
    model.train()
    mask_ratio = token_mask_ratio(epoch, EPOCHS)
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    running = 0.0
    n_batches = 0

    for x, y in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # y = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-6)

        # mask non-predictable
        predictable_mask = (x[:, :, 0] != 0).float()

        # 🔥 token masking
        x = apply_token_mask(x, mask_ratio)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=AMP):
            y_hat, confi = model(x)
            masked_confi = confi * predictable_mask
            masked_confi = masked_confi / masked_confi.sum(dim=1, keepdim=True)

            loss_mse = ((y_hat - y).pow(2) * predictable_mask).sum() / (predictable_mask.sum() + 1e-8)
            loss_exp = torch.sum((y * masked_confi), dim=1).mean()
            loss_var = torch.sum((y * masked_confi), dim=1).var()

            loss = loss_mse * 0.0 - loss_exp * EXP_LOSS_WEIGHT + loss_var * VAR_LOSS_WEIGHT

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

    train_loader = DataLoader(TimeIndexDataset(X_train, Y_train),
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TimeIndexDataset(X_val, Y_val),
                            batch_size=TEST_BATCH_SIZE)
    test_loader = DataLoader(TimeIndexDataset(X_test, Y_test),
                             batch_size=TEST_BATCH_SIZE)

    csv_rows = []
    best_models = []
    best_corrs = []

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

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR * 0.05
        )

        scaler = torch.amp.GradScaler("cuda", enabled=AMP)

        best_confi_exp = -1.0
        best_confi_var = -1.0
        ckpt_path = OUT_DIR / f"best_model_seed{seed}.pt"

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, scheduler, epoch
            )
            val_rmse, val_mae, val_exp, val_var, val_confi_rmse, val_confi_mae, val_confi_exp, val_confi_var = evaluate(model, val_loader)
            # print(f"val_rmse={val_rmse:.4f} " 
            #       f"val_mae={val_mae:.4f} "
            #       f"val_exp={val_exp:.4f} "
            #       f"val_var={val_var:.4f} "
            #       f"val_confi_rmse={val_confi_rmse:.4f} " 
            #       f"val_confi_mae={val_confi_mae:.4f} "
            #       f"val_confi_exp={val_confi_exp:.4f} "
            #       f"val_confi_var={val_confi_var:.4f} ")

            csv_rows.append([
                seed, epoch, train_loss,
                val_rmse, val_mae, val_exp, val_var,
                val_confi_rmse, val_confi_mae, val_confi_exp, val_confi_var,
                optimizer.param_groups[0]["lr"]
            ])

            if val_confi_exp * EXP_LOSS_WEIGHT + val_confi_var * VAR_LOSS_WEIGHT \
                  > best_confi_exp * EXP_LOSS_WEIGHT + best_confi_var * VAR_LOSS_WEIGHT:
                best_confi_exp = val_confi_exp
                best_confi_var = val_confi_var
                torch.save(model.state_dict(), ckpt_path)
                # print(f"Saved best model to {ckpt_path} with val_confi_corr={val_confi_corr} and val_y_hat_confi_variance={val_y_hat_confi_variance}")

        model.load_state_dict(torch.load(ckpt_path))
        best_models.append(model)
        best_corrs.append(best_confi_exp * EXP_LOSS_WEIGHT + best_confi_var * VAR_LOSS_WEIGHT)
        print(f"seed={seed} best_confi_exp={best_confi_exp} best_confi_var={best_confi_var}")

    # =========================
    # Save CSV
    # =========================
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "epoch", "train_loss",
            "val_rmse", "val_mae", "val_exp", "val_var",
            "val_confi_rmse", "val_confi_mae", "val_confi_exp", "val_confi_var", "lr"
        ])
        writer.writerows(csv_rows)

    # =========================
    # Plot
    # =========================
    import pandas as pd
    df = pd.read_csv(LOG_CSV)

    for col in ["train_loss", "val_rmse", "val_mae", "val_exp", "val_var", "val_confi_rmse", "val_confi_mae", "val_confi_exp", "val_confi_var"]:
        plt.figure()
        for seed in SEEDS:
            d = df[df.seed == seed]
            plt.plot(d.epoch, d[col], label=f"seed={seed}")
        plt.legend()
        plt.title(col)
        plt.savefig(OUT_DIR / f"{col}.png")
        plt.close()

    # =========================
    # Ensemble Test
    # =========================
    print("Ensemble weights (val_corr):", best_corrs)

    test_rmse, test_mae, test_exp, test_var, test_confi_rmse, test_confi_mae, test_confi_exp, test_confi_var = evaluate_ensemble(
        best_models,
        best_corrs,
        test_loader
    )

    print(f"[Weighted Ensemble Test] "
        f"rmse={test_rmse:.6f} "
        f"mae={test_mae:.6f} "
        f"exp={test_exp:.6f} "
        f"var={test_var:.6f} "
        f"confi_rmse={test_confi_rmse:.4f} "
        f"confi_mae={test_confi_mae:.4f} "
        f"confi_exp={test_confi_exp:.4f} "
        f"confi_var={test_confi_var:.4f} ")

if __name__ == "__main__":
    main()
