import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# Config
# =========================
BASE_DIR = Path("backtest")
INFER_DIR = BASE_DIR / "inference_results"
REFINED_DIR = Path("refined_data")
OUT_DIR = BASE_DIR / "backtest_results"

OUT_DIR.mkdir(parents=True, exist_ok=True)

START_SEED_MONEY = 1_000_000.0   # 시작 자금 (원, 단위 자유)
TOP_K = 3


# =========================
# Load refined data cache
# =========================
def load_refined_data(ticker):
    path = REFINED_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing refined_data: {ticker}")
    df = pd.read_csv(path)
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.set_index("날짜").sort_index()
    return df


# =========================
# Main Backtest
# =========================
def main():
    inference_files = sorted(INFER_DIR.glob("*.csv"))

    seed_money = START_SEED_MONEY
    equity_curve = []   # (date, seed_money)

    refined_cache = {}

    print("===== Backtest start =====")

    for infer_path in tqdm(inference_files):
        infer_df = pd.read_csv(infer_path, index_col=0)
        infer_df.index = pd.to_datetime(infer_df.index)

        for date, row in infer_df.iterrows():
            scores = row.sort_values(ascending=False)

            selected = []
            for ticker in scores.index:
                if ticker not in refined_cache:
                    refined_cache[ticker] = load_refined_data(ticker)

                ref_df = refined_cache[ticker]

                if date not in ref_df.index:
                    print(f"Missing date: {ticker} {date}")
                    continue

                if ref_df.loc[date, "predictable"] == 1.0:
                    selected.append(ticker)

                if len(selected) == TOP_K:
                    break

            # 선택 종목 부족하면 skip (현금 보유)
            if len(selected) < TOP_K:
                equity_curve.append((date, seed_money))
                continue

            alloc = seed_money / TOP_K
            next_seed_money = 0.0

            for ticker in selected:
                ref_df = refined_cache[ticker]

                try:
                    today_close = ref_df.loc[date, "종가"]
                    next_date = ref_df.index[ref_df.index.get_loc(date) + 1]
                    next_close = ref_df.loc[next_date, "종가"]
                except (KeyError, IndexError):
                    # 다음 영업일 없으면 해당 포지션 유지 불가
                    next_seed_money += alloc
                    continue

                # =========================
                # 수익률 계산
                # =========================
                ret = next_close / today_close
                next_seed_money += alloc * ret

            seed_money = next_seed_money
            equity_curve.append((date, seed_money))

    # =========================
    # Save results
    # =========================
    equity_df = pd.DataFrame(equity_curve, columns=["date", "equity"])
    equity_df = equity_df.drop_duplicates(subset="date")
    equity_df = equity_df.set_index("date").sort_index()

    equity_df.to_csv(OUT_DIR / "equity_curve.csv")

    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(10, 6))
    plt.plot(equity_df.index, equity_df["equity"])
    plt.yscale("log")   # ✅ y축 로그 변환
    plt.title("Backtest Equity Curve (Log Scale)")
    plt.xlabel("Date")
    plt.ylabel("Seed Money (log)")
    plt.grid(True, which="both")  # 로그 스케일에서는 both 추천
    plt.tight_layout()
    plt.savefig(OUT_DIR / "equity_curve_log.png")
    plt.close()

    print("===== Backtest finished =====")
    print(f"Final seed money: {seed_money:,.0f}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()