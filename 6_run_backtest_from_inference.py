import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


# =========================
# Config
# =========================
BASE_DIR = Path("backtest")
INFER_DIR = BASE_DIR / "inference_results"
REFINED_DIR = Path("refined_data")
OUT_DIR = BASE_DIR / "backtest_results"
TEST_VAL_LST_PATH = BASE_DIR / "regression_runs/test_val_lst.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

START_SEED_MONEY = 1_000_000.0   # 시작 자금 (원, 단위 자유)
TOP_K = 5                        # 하루에 매매할 종목 개수
PRE_SELECTED_TOLERANCE = 0.0     # 전날에 매수한 종목을 그대로 유지할지를 결정
BUY_THRESHOLD = 0.03             # 예측값이 1차적으로 이 값을 넘어야 매수
HALT_THRESHOLD = 0.0            # 한 달의 수익률이 이보다 낮으면 그 달은 skip

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
    test_val_lst = pickle.load(open(TEST_VAL_LST_PATH, "rb"))

    seed_money = START_SEED_MONEY
    equity_curve = []   # (date, seed_money)

    refined_cache = {}
    pre_selected = []

    print("===== Backtest start =====")

    for infer_path in tqdm(inference_files):
        infer_df = pd.read_csv(infer_path, index_col=0)
        infer_df.index = pd.to_datetime(infer_df.index)

        return_record = []
        halt_flag = False

        for date, row in infer_df.iterrows():
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
    # 2차원 리스트에서 index 7 값만 추출 → 1차원 시계열
    test_series = []
    for row in test_val_lst:
        test_series += [row[6]] * 20
    test_series += [test_val_lst[-1][6]] * 32

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Equity (왼쪽 y축, 로그) ---
    ax1.plot(equity_df.index, equity_df["equity"], label="Equity")
    ax1.set_yscale("log")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Equity (log)")

    # --- Test value (오른쪽 y축) ---
    ax2 = ax1.twinx()
    ax2.plot(equity_df.index, test_series, linestyle="--", label="Test Value")
    ax2.set_ylabel("Test Value")

    ax1.grid(True, which="both")

    # 범례 합치기
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.title("Backtest Equity Curve (Log Scale)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "equity_curve_log.png")
    plt.close()

    print("===== Backtest finished =====")
    print(f"Final seed money: {seed_money:,.0f}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()