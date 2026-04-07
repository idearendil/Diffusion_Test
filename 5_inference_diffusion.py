import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import calendar
import numpy as np
from scipy.stats import norm

from utils import list_tickers, make_t_seq, cosine_beta_schedule, y0_from_v_yt, eps_from_v_yt
from model import DiffusionTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Paths
# =========================
BASE_DIR = Path("backtest")
TENSOR_ROOT = BASE_DIR / "tensor_data"
MODEL_ROOT  = BASE_DIR / "models"
OUT_ROOT    = BASE_DIR / "diffusion_results"
REFINED_DIR = Path("refined_data")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# Diffusion config (train과 동일해야 함)
# =========================
T_STEPS = 1000
SAMPLE_STEPS = 100
K_SAMPLES = 100   # ⭐ 핵심
DDIM_ETA = 0.0


# =========================
# Fixed Gaussian samples
# =========================
def make_fixed_gaussian_samples(K, shape, device):
    """
    Stratified Gaussian sampling
    """
    u = (np.arange(K) + 0.5) / K
    z_vals = norm.ppf(u)  # inverse CDF

    z_vals = torch.tensor(z_vals, dtype=torch.float32, device=device)

    z = z_vals.view(K, 1, 1).expand(K, *shape)
    return z


# =========================
# DDIM sampling
# =========================
@torch.no_grad()
def ddim_sample_y0(model, x, diffusion, t_seq, z_fixed, eta=0.0):
    model.eval()
    alpha_bar = diffusion["alpha_bar"]

    B, N, _ = x.shape
    y = torch.randn(B, N, device=x.device)

    for i, t_i in enumerate(t_seq):
        t = torch.full((B,), int(t_i.item()), device=x.device, dtype=torch.long)

        tokens = torch.cat([y.unsqueeze(-1), x], dim=-1)
        v_pred = model(tokens, t)

        ab_t_scalar = alpha_bar[t_i]
        ab_t = torch.full((B,), float(ab_t_scalar.item()), device=x.device, dtype=torch.float32)

        y0_hat = y0_from_v_yt(v_pred, y, ab_t)
        eps_pred = eps_from_v_yt(v_pred, y, ab_t)

        if i == len(t_seq) - 1:
            y = y0_hat
            break

        t_prev = t_seq[i + 1]
        ab_prev_scalar = alpha_bar[t_prev]
        sqrt_ab_prev = torch.sqrt(ab_prev_scalar)

        if eta > 0.0:
            sigma = eta * torch.sqrt((1 - ab_prev_scalar) / (1 - ab_t_scalar)) * torch.sqrt(1 - ab_t_scalar / ab_prev_scalar)
            z = z_fixed
        else:
            sigma = 0.0
            z = 0.0

        c = torch.sqrt(torch.clamp(1.0 - ab_prev_scalar - sigma**2, min=0.0))
        y = sqrt_ab_prev * y0_hat + c * eps_pred + sigma * z

    return y

# =========================
# K sampling → mean / std
# =========================
@torch.no_grad()
def sample_k_all(model, X, diffusion):
    t_seq = make_t_seq(T_STEPS, SAMPLE_STEPS, X.device)

    z_fixed = make_fixed_gaussian_samples(
        K_SAMPLES,
        (X.shape[0], X.shape[1]),
        X.device
    )

    samples = []
    for i in tqdm(range(K_SAMPLES), desc="Sampling"):
        y0_hat = ddim_sample_y0(model, X, diffusion, t_seq, z_fixed[i], eta=DDIM_ETA)
        samples.append(y0_hat)

    S = torch.stack(samples, dim=0)  # [K, T, N]
    return S


# =========================
# Trading days
# =========================
def load_trading_days(start_date: str, end_date: str):
    ref = pd.read_csv(REFINED_DIR / "000020.csv")
    ref["날짜"] = pd.to_datetime(ref["날짜"])

    mask = (ref["날짜"] >= start_date) & (ref["날짜"] <= end_date)
    return ref.loc[mask, "날짜"].dt.strftime("%Y-%m-%d").tolist()


# =========================
# Load tensor
# =========================
def load_test_tensor(split_dir: Path, tickers):
    x_list = []

    for tkr in tickers:
        x = torch.load(split_dir / f"{tkr}_x.pt", map_location="cpu")
        x_list.append(x.float())

    X = torch.stack(x_list, dim=0).transpose(0, 1).contiguous()  # [T,N,F]
    return X


# =========================
# Main
# =========================
def main():
    date_dirs = sorted([d for d in TENSOR_ROOT.iterdir() if d.is_dir()])

    for date_dir in date_dirs:
        date = date_dir.name
        print(f"\n===== Diffusion Inference {date} =====")

        out_csv = OUT_ROOT / f"{date}.csv"
        if out_csv.exists():
            print(f"[SKIP] {date}")
            continue

        test_dir = date_dir / "test"
        model_dir = MODEL_ROOT / date
        ckpt_path = model_dir / "best_model.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing model: {ckpt_path}")

        tickers = list_tickers(test_dir)
        tickers.sort()

        # -------------------------
        # Load data
        # -------------------------
        X = load_test_tensor(test_dir, tickers).to(DEVICE)
        T, N, F = X.shape

        # -------------------------
        # Load model
        # -------------------------
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        model = DiffusionTransformer(
            n_tokens=N,
            in_dim=F + 1,
            d_model=ckpt["config"]["D_MODEL"],
            n_head=ckpt["config"]["N_HEAD"],
            n_layers=ckpt["config"]["N_LAYERS"],
            d_ff=ckpt["config"]["D_FF"],
            dropout=ckpt["config"]["DROPOUT"],
        ).to(DEVICE)

        model.load_state_dict(ckpt["model_state"])

        # diffusion params
        betas = cosine_beta_schedule(T_STEPS, s=0.008, max_beta=0.02).to(DEVICE)
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

        # -------------------------
        # Sampling
        # -------------------------
        S = sample_k_all(model, X, diffusion)  # [K,T,N]
        S = S.cpu().numpy()

        # mask 반영
        mask = (X[:, :, 0] != 0).cpu().numpy()
        for k in range(K_SAMPLES):
            S[k][mask == 0] = -100.0

        # -------------------------
        # Trading days
        # -------------------------
        year  = int(date[:4])
        month = int(date[5:7])
        last_day = calendar.monthrange(year, month)[1]

        trading_days = load_trading_days(
            f"{year}-{month:02d}-01",
            f"{year}-{month:02d}-{last_day}"
        )
        if year == 2025 and month == 12:
            trading_days = trading_days[:-1]

        if len(trading_days) != T:
            raise ValueError(f"{date}: trading_days({len(trading_days)}) != T({T})")

        # -------------------------
        # DataFrame 생성
        # -------------------------
        rows = []
        for k in range(K_SAMPLES):
            df_k = pd.DataFrame(
                S[k], 
                index=trading_days, 
                columns=tickers
            )
            df_k["sample_id"] = k
            rows.append(df_k)

        df_all = pd.concat(rows)

        cols = ["sample_id"] + tickers
        df_all = df_all[cols]
        df_all.index.name = "date"

        # -------------------------
        # 저장 (⭐ 파일 2개)
        # -------------------------
        out_csv = OUT_ROOT / f"{date}.csv"
        df_all.to_csv(out_csv)

        print(f"[OK] saved samples → {out_csv}")

if __name__ == "__main__":
    main()