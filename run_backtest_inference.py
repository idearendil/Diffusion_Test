import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import calendar
import numpy as np

from utils import list_tickers
from model_regression import RegressionLSTMTransformer   # ← 네가 말한 

from utils import SEQ_LEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Paths
# =========================
BASE_DIR = Path("backtest")
TENSOR_ROOT = BASE_DIR / "tensor_data"
MODEL_ROOT  = BASE_DIR / "regression_runs"
OUT_ROOT    = BASE_DIR / "inference_results"
REFINED_DIR = Path("refined_data")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(3))


# =========================
# Load trading days
# =========================
def load_trading_days(start_date: str, end_date: str):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    ref = pd.read_csv(REFINED_DIR / "000020.csv")
    ref["날짜"] = pd.to_datetime(ref["날짜"])
    mask = (ref["날짜"] >= start_date) & (ref["날짜"] <= end_date)
    return ref.loc[mask, "날짜"].dt.strftime("%Y-%m-%d").tolist()


# =========================
# Load test tensors (concat tickers)
# =========================
def load_test_tensor(split_dir: Path, tickers):
    x_list = []

    for tkr in tickers:
        x_path = split_dir / f"{tkr}_x.pt"
        if not x_path.exists():
            raise FileNotFoundError(f"Missing x tensor: {tkr}")

        x = torch.load(x_path, map_location="cpu")  # [T, F]
        if x.ndim != 2:
            raise ValueError(f"{tkr}: bad x shape {x.shape}")

        x_list.append(x.float())

    # [N, T, F] → [T, N, F]
    X = torch.stack(x_list, dim=0).transpose(0, 1).contiguous()
    return X


def load_ensemble_weights(date):
    """
    date: '2020-01-01'
    return: dict {seed: weight}
    """
    metrics_path = BASE_DIR / "regression_runs" / date / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv for {date}")

    df = pd.read_csv(metrics_path)

    # seed별 val_exp 최대값
    weights = (
        df.groupby("seed")["val_exp"]
        .max()
        .to_dict()
    )

    # 음수 방지 + 정규화
    w = np.array(list(weights.values()), dtype=np.float64)
    w = np.clip(w, 0.0, None)
    w = w / (w.sum() + 1e-8)

    return dict(zip(weights.keys(), w))


# =========================
# Inference (ensemble mean)
# =========================
@torch.no_grad()
def run_ensemble(models, X, weights):
    preds = []

    for model_id, model in enumerate(models):
        model.eval()
        _, y_hat = model(X)   # [T, N]
        preds.append(y_hat * weights[model_id])

    return torch.stack(preds).sum(dim=0)  # [T, N]


def make_lstm_sequences(x, seq_len=10):
    """
    x: [T, N, F]
    return: [T-seq_len+1, seq_len, N, F]
    """

    T, N, F = x.shape

    if T < seq_len:
        raise ValueError("T must be >= seq_len")

    xs = []
    for i in range(T - seq_len + 1):
        xs.append(x[i:i + seq_len])   # [seq_len, N, F]

    return torch.stack(xs, dim=0)     # [T-seq_len+1, seq_len, N, F]


# =========================
# Main
# =========================
def main():
    date_dirs = sorted([d for d in TENSOR_ROOT.iterdir() if d.is_dir()])

    for date_dir in date_dirs:
        date = date_dir.name
        print(f"\n===== Inference {date} =====")

        out_csv = OUT_ROOT / f"{date}.csv"
        if out_csv.exists():
            print(f"[SKIP] {date} already inferred")
            continue

        test_dir = date_dir / "test"
        model_dir = MODEL_ROOT / date

        tickers = list_tickers(test_dir)
        tickers.sort()

        # -------------------------
        # Load tensors
        # -------------------------
        X = load_test_tensor(test_dir, tickers)
        T, N, F = X.shape
        X = make_lstm_sequences(X, SEQ_LEN)  # [T-seq_len+1, seq_len, N, F]
        X = X.to(DEVICE)

        # -------------------------
        # Load models
        # -------------------------
        ensemble_weights = load_ensemble_weights(date)

        models = []
        for seed in SEEDS:
            ckpt = model_dir / f"best_model_seed{seed}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing model: {ckpt}")

            model = RegressionLSTMTransformer(
                n_tokens=N,   # ⚠ token dim 변경
                in_dim=F,     # ⚠ feature dim 변경
                d_model=128,
                n_head=8,
                n_layers=3,
                d_ff=256,
                lstm_layers=1,
                dropout=0.1,
            ).to(DEVICE)


            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            models.append(model)

        # -------------------------
        # Inference
        # -------------------------
        Y_hat = run_ensemble(models, X, ensemble_weights)  # [T, N]
        Y_hat = Y_hat.cpu().numpy()

        # -------------------------
        # Trading days
        # -------------------------
        year  = int(date[:4])
        month = int(date[5:7])
        last_day = calendar.monthrange(year, month)[1]
        start = f"{year:04d}-{month:02d}-01"
        end   = f"{year:04d}-{month:02d}-{last_day:02d}"
        trading_days = load_trading_days(start, end)
        if year == 2025 and month == 12:
            trading_days = trading_days[:-1]

        if len(trading_days) != T - SEQ_LEN + 1:
            raise ValueError(
                f"{date}: trading_days({len(trading_days)}) != T({T - SEQ_LEN + 1})"
            )

        # -------------------------
        # Save CSV
        # -------------------------
        df = pd.DataFrame(Y_hat, index=trading_days, columns=tickers)
        df.index.name = "date"
        df.to_csv(out_csv)

        print(f"[OK] saved → {out_csv}")


if __name__ == "__main__":
    main()