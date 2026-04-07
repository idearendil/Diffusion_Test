import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch


# =========================
# Config
# =========================
BASE_DIR = Path("backtest")
INFER_DIR = BASE_DIR / "diffusion_results"
REFINED_DIR = Path("refined_data")
OUT_DIR = BASE_DIR / "backtest_results"

OUT_DIR.mkdir(parents=True, exist_ok=True)

START_SEED_MONEY = 1_000_000.0   # 시작 자금 (원, 단위 자유)
TOP_K = 3                        # 하루에 매매할 종목 개수
PRE_SELECTED_TOLERANCE = 0.0     # 전날에 매수한 종목을 그대로 유지할지를 결정
BUY_THRESHOLD = 0.02             # 예측값이 1차적으로 이 값을 넘어야 매수
HALT_THRESHOLD = 0.0            # 한 달의 수익률이 이보다 낮으면 그 달은 skip
Z_LINEAR_THRESHOLD = [0.6, 0.46]  # 0.6 => 72.5%, 0.46 => 67.5%

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
    pre_selected = []

    print("===== Backtest start =====")

    z_index = -1
    for file_path in tqdm(inference_files):
        z_index += 1
        inference_df = pd.read_csv(file_path, index_col=0)
        inference_df.index = pd.to_datetime(inference_df.index)

        grouped = inference_df.groupby(inference_df.index)
        mean_df = grouped.mean()
        std_df  = grouped.std()
        if "sample_id" in mean_df.columns:
            mean_df = mean_df.drop(columns=["sample_id"])
            std_df  = std_df.drop(columns=["sample_id"])
        
        z_rate = z_index / (len(inference_files) - 1)
        z_value = Z_LINEAR_THRESHOLD[0] * (1.0 - z_rate) + Z_LINEAR_THRESHOLD[1] * z_rate
        score_df = mean_df - z_value * std_df  # 0.2% 이상 상승할 가능성이 70% 이상

        return_record = []
        halt_flag = False

        for date, row in score_df.iterrows():
            scores = row.sort_values(ascending=False)

            selected = []
            maintained = []
            tolerance_num = int(len(scores.index) * PRE_SELECTED_TOLERANCE)
            tolerance_cnt = 0
            for ticker in scores.index:
                tolerance_cnt += 1

                if ticker not in refined_cache:
                    refined_cache[ticker] = load_refined_data(ticker)
                ref_df = refined_cache[ticker]

                if date not in ref_df.index:
                    print(f"Missing date: {ticker} {date}")
                    continue

                if ref_df.loc[date, "predictable"] == 1.0 and scores[ticker] > BUY_THRESHOLD:
                    if ticker in pre_selected and tolerance_cnt <= tolerance_num:
                        maintained.append(ticker)
                    else:
                        selected.append(ticker)

            final_selected = maintained
            for ticker in selected:
                if len(final_selected) >= TOP_K:
                    break
                final_selected.append(ticker)

            # 선택 종목 부족하면 skip (현금 보유)
            if len(final_selected) < TOP_K:
                equity_curve.append((date, seed_money))
                continue

            # 해당 모델로 너무 많이 잃었다면 stop loss
            if not halt_flag:
                result = 1.0
                for row in return_record:
                    row_mean = sum(row) / len(row)
                    result *= row_mean
                if result < HALT_THRESHOLD:
                    halt_flag = True
            if halt_flag:
                equity_curve.append((date, seed_money))
                continue
            else:
                return_record.append([])

            alloc = seed_money / TOP_K
            next_seed_money = 0.0

            for ticker in final_selected:
                ref_df = refined_cache[ticker]

                try:
                    today_close = ref_df.loc[date, "종가"]
                    next_date = ref_df.index[ref_df.index.get_loc(date) + 1]
                    next_close = ref_df.loc[next_date, "종가"]
                    next_low = ref_df.loc[next_date, "저가"]
                except (KeyError, IndexError):
                    # 다음 영업일 없으면 해당 포지션 유지 불가
                    next_seed_money += alloc
                    continue

                if next_low < today_close * 0.9:
                    next_close = today_close * 0.9

                # =========================
                # 수익률 계산
                # =========================
                # print(ticker, (next_close / today_close - 1) * 100)
                if ticker in pre_selected:
                    bought_stocks = alloc // (today_close * (1 - 0.0001) * (1 - 0.002))     # 이전에 샀던 stock 개수 그대로
                    leftover = alloc - bought_stocks * (today_close * (1 - 0.0001) * (1 - 0.002))
                    next_seed_money += (bought_stocks * next_close * (1 - 0.0001) * (1 - 0.002) + leftover)    # 매도: 매매수수료 + 거래세 적용
                else:
                    bought_stocks = alloc // (today_close * (1 + 0.0001))     # 매수: 매매수수료 적용
                    leftover = alloc - bought_stocks * today_close * (1 + 0.0001)
                    next_seed_money += (bought_stocks * next_close * (1 - 0.0001) * (1 - 0.002) + leftover)    # 매도: 매매수수료 + 거래세 적용

                return_record[-1].append((next_close / today_close) - 0.0022)

            seed_money = next_seed_money
            equity_curve.append((date, seed_money))
            pre_selected = final_selected

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
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Equity (왼쪽 y축, 로그) ---
    ax1.plot(equity_df.index, equity_df["equity"], label="Equity")
    ax1.set_yscale("log")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Equity (log)")

    ax1.grid(True, which="both")

    # 범례 합치기
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1)

    plt.title("Backtest Equity Curve (Log Scale)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "equity_curve_log.png")
    plt.close()

    print("===== Backtest finished =====")
    print(f"Final seed money: {seed_money:,.0f}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()