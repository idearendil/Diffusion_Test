import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# === 경로 & 컬럼 설정 ===
REFINED_DIR = Path("refined_data")

BACKTEST_ROOT = Path("backtest") / "tensor_data"
BACKTEST_ROOT.mkdir(parents=True, exist_ok=True)

DATE_COL = "날짜"
OPEN_COL = "시가"
HIGH_COL = "고가"
LOW_COL = "저가"
CLOSE_COL = "종가"
VOLUME_COL = "거래량"

LABEL_COL = "label1"
CHANGE_RATE_COL = "ret_pct"  # 종가 기준 일간 수익률(%)

TEST_START = pd.Timestamp("2020-01-01")
TEST_END   = pd.Timestamp("2025-12-31")


# =========================
# Utils
# =========================
def is_volume_related(col: str) -> bool:
    return col.startswith("vol_") or ("거래량" in col)


def split_4_1(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """앞에서부터 순서대로 4:1 비율 split (train/val)."""
    n = len(df)
    train_end = (n * 4) // 5

    # 안전장치
    train_end = max(train_end, 1)
    train_end = min(train_end, n - 1)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df_shuffled.iloc[:train_end].copy()
    val_df = df_shuffled.iloc[train_end:].copy()
    return train_df, val_df


def split_11_1(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """앞에서부터 순서대로 11:1 비율 split (train/val)."""
    n = len(df)
    train_end = (n * 11) // 12

    # 안전장치
    train_end = max(train_end, 1)
    train_end = min(train_end, n - 1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:].copy()
    return train_df, val_df


def clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """(label, vol_* 제외) feature를 -10~10 클리핑"""
    df = df.copy()
    feature_cols = [c for c in df.columns if c != LABEL_COL and c != DATE_COL]
    feature_cols = [c for c in feature_cols if c != "day_of_week" and c != "day_of_month"]
    clip_cols = [c for c in feature_cols if not is_volume_related(c)]
    if clip_cols:
        df[clip_cols] = df[clip_cols].clip(-10.0, 10.0)
    return df


def standardize_df(df: pd.DataFrame, feature_cols: List[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    df = df.copy()

    # 컬럼 순서 맞춤
    df = df[[*feature_cols, LABEL_COL]]

    x = (df[feature_cols] - mean) / std
    x = x.fillna(0.0)

    x['predictable'] = np.where(x['predictable'] > 0, 1, 0)

    out = pd.concat([x, df[[LABEL_COL]].astype(np.float32)], axis=1)
    return out


def save_tensors(df: pd.DataFrame, feature_cols: List[str], out_dir: Path, ticker: str) -> None:
    x = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    torch.save(x_t, out_dir / f"{ticker}_x.pt")
    torch.save(y_t, out_dir / f"{ticker}_y.pt")


# =========================
# Feature / Label
# =========================
def add_features_and_labels(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """단일 종목(df)에 대해 feature / label 생성."""
    df = df.copy()
    df = df.sort_values(DATE_COL)

    # day_of_month: 1~31 (월마다 일수가 다르므로 "해당 월의 일수"로 스케일링하는 게 보통 더 좋음)
    dom = df[DATE_COL].dt.day.astype(np.float32)                  # 1..31
    dim = df[DATE_COL].dt.days_in_month.astype(np.float32)        # 28..31

    dom_phase = 2.0 * np.pi * (dom - 1.0) / dim                   # 0..2pi
    df["dom_sin"] = np.sin(dom_phase)
    df["dom_cos"] = np.cos(dom_phase)

    # day_of_week: pandas dayofweek = 0(Mon) .. 6(Sun)
    dow = df[DATE_COL].dt.dayofweek.astype(np.float32)            # 0..6
    dow_phase = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_phase)
    df["dow_cos"] = np.cos(dow_phase)

    # 0) 가격 관련 파생
    prev_close = df[CLOSE_COL].shift(1)
    df["gap"] = (df[OPEN_COL] - prev_close) / (prev_close + 1.0)
    df["high_low_spread"] = np.log((df[HIGH_COL] - df[LOW_COL]) / (df[CLOSE_COL] + 1.0) + 1e-10)

    # refined_data에는 등락률이 없으니 종가 기반으로 생성(%)  -> 원래의 CHANGE_RATE_COL 대체
    df[CHANGE_RATE_COL] = (df[CLOSE_COL] / (df[CLOSE_COL].shift(1) + 1e-12) - 1.0) * 100.0

    # 1) 거래량 = 거래량 * 시가 (원 코드 유지)
    # df[VOLUME_COL] = df[VOLUME_COL] * df[OPEN_COL]
    df["vol_log"] = np.log(df[VOLUME_COL] * df[OPEN_COL] + 1.0)
    df["vol_log_ma_5"] = df["vol_log"].rolling(window=5, min_periods=3).mean()
    df["vol_log_ma_20"] = df["vol_log"].rolling(window=20, min_periods=6).mean()
    df["vol_log_ma_60"] = df["vol_log"].rolling(window=60, min_periods=20).mean()
    df["vol_ma_5"] = df[VOLUME_COL].rolling(window=5, min_periods=3).mean()
    df["vol_ma_20"] = df[VOLUME_COL].rolling(window=20, min_periods=6).mean()
    df["vol_ma_60"] = df[VOLUME_COL].rolling(window=60, min_periods=20).mean()
    df["vol_ratio_5"] = np.log(df[VOLUME_COL] / (df["vol_ma_5"] + 1.0) + 1.0)
    df["vol_ratio_20"] = np.log(df[VOLUME_COL] / (df["vol_ma_20"] + 1.0) + 1.0)
    df["vol_ratio_60"] = np.log(df[VOLUME_COL] / (df["vol_ma_60"] + 1.0) + 1.0)
    df["vol_diff"] = df[VOLUME_COL] / (df[VOLUME_COL].shift(1) + 1.0)
    df["vol_diff"] = np.log(np.clip(df["vol_diff"].values, 0.0, 10.0) + 1.0)
    df["vol_diff_ma_5"] = df["vol_diff"].diff().rolling(window=5, min_periods=3).mean()
    df["vol_diff_ma_20"] = df["vol_diff"].diff().rolling(window=20, min_periods=6).mean()
    df["vol_diff_ma_60"] = df["vol_diff"].diff().rolling(window=60, min_periods=20).mean()

    # 8) 과거 수익률 기반 평균/변동성
    for w in [5, 10, 20, 60]:
        minp = max(3, w // 3)
        df[f"ret_mean_{w}"] = df[CHANGE_RATE_COL].rolling(window=w, min_periods=minp).mean()
        df[f"ret_std_{w}"] = np.log(df[CHANGE_RATE_COL].rolling(window=w, min_periods=minp).std() + 1.0)

    df[CHANGE_RATE_COL + "_2"] = df[CHANGE_RATE_COL].shift(1)
    df[CHANGE_RATE_COL + "_3"] = df[CHANGE_RATE_COL].shift(2)
    df[CHANGE_RATE_COL + "_4"] = df[CHANGE_RATE_COL].shift(3)
    df[CHANGE_RATE_COL + "_5"] = df[CHANGE_RATE_COL].shift(4)

    # 9) 갭의 N일 평균
    for w in [5, 10, 20]:
        minp = max(3, w // 3)
        df[f"gap_mean_{w}"] = df["gap"].rolling(window=w, min_periods=minp).mean()

    # 10) 가격 레벨 장단기 평균/최고/최저 대비
    for w in [5, 20, 60]:
        minp = max(3, w // 3)
        roll_mean = df[CLOSE_COL].rolling(window=w, min_periods=minp).mean()
        roll_max = df[CLOSE_COL].rolling(window=w, min_periods=minp).max()
        roll_min = df[CLOSE_COL].rolling(window=w, min_periods=minp).min()
        df[f"close_to_ma_{w}"] = df[CLOSE_COL] / (roll_mean + 1.0)
        df[f"close_to_max_{w}"] = np.log(-(df[CLOSE_COL] / (roll_max + 1.0)) + 1.0 + 1e-10)
        df[f"close_to_min_{w}"] = np.log(df[CLOSE_COL] / (roll_min + 1.0) + 1e-10)

    # 모멘텀
    log_close = np.log(df[CLOSE_COL] + 1.0)
    df["mom_5"] = log_close - log_close.shift(5)
    df["mom_20"] = log_close - log_close.shift(20)

    # True Range
    prev_close = df[CLOSE_COL].shift(1)
    tr = np.maximum(
        df[HIGH_COL] - df[LOW_COL],
        np.maximum(
            (df[HIGH_COL] - prev_close).abs(),
            (df[LOW_COL] - prev_close).abs()
        )
    )
    for w in [5, 14, 20]:
        df[f"atr_{w}"] = tr.rolling(window=w, min_periods=w//2).mean()
        df[f"atr_ratio_{w}"] = np.log(df[f"atr_{w}"] / (df[CLOSE_COL] + 1.0) + 1e-10)

    # Candle Body Ratio / Upper/Lower Wick Ratio
    body = (df[CLOSE_COL] - df[OPEN_COL]).abs()
    upper_wick = df[HIGH_COL] - np.maximum(df[CLOSE_COL], df[OPEN_COL])
    lower_wick = np.minimum(df[CLOSE_COL], df[OPEN_COL]) - df[LOW_COL]
    range_ = (df[HIGH_COL] - df[LOW_COL]) + 1e-6
    df["candle_body_ratio"] = np.log(body / range_ + 1e-10)
    df["upper_wick_ratio"] = np.log(np.clip(upper_wick / range_, 0.0, 10.0) + 1.0)
    df["lower_wick_ratio"] = np.log(lower_wick / range_ + 1e-10)
    df["bullish"] = (df[CLOSE_COL] > df[OPEN_COL]).astype(np.float32)

    # MACD
    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()
    ema12 = ema(df[CLOSE_COL], 12)
    ema26 = ema(df[CLOSE_COL], 26)
    df["macd"] = ema12 - ema26
    df["macd_signal"] = ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI
    delta = df[CLOSE_COL].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    ma20 = df[CLOSE_COL].rolling(20, min_periods=10).mean()
    std20 = df[CLOSE_COL].rolling(20, min_periods=10).std()
    df["bb_upper"] = (ma20 + 2 * std20) / (df[CLOSE_COL] + 1.0)
    df["bb_lower"] = (ma20 - 2 * std20) / (df[CLOSE_COL] + 1.0)
    df["bb_width"] = np.log((2 * std20) / (ma20 + 1.0) + 1e-10)

    # OBV
    direction = np.sign(df[CLOSE_COL].diff()).fillna(0)
    df["obv"] = (direction * df[VOLUME_COL]).cumsum()
    df["obv_ma_20"] = df["obv"].rolling(20, min_periods=10).mean()

    # VPT
    df["vpt"] = ((df[CLOSE_COL] - df[CLOSE_COL].shift(1)) /
             (df[CLOSE_COL].shift(1) + 1e-6) * df[VOLUME_COL]).cumsum()

    # ===== Label =====
    df[LABEL_COL] = (df[CLOSE_COL].shift(-1) / df[CLOSE_COL] - 1) * 10.0

    # 다음날 없는 마지막 행 + 이전 행들이 적은 첫 부분 행들 제거
    df = df.dropna()

    return df

def prepare_full_dfs() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    모든 ticker CSV를 한 번 읽어서 feature/label 만든 뒤
    (DATE_COL 유지) full_df[ticker]로 보관.
    + feature 컬럼 전역 순서(feature_cols_order) 생성.
    """
    csv_paths = sorted(REFINED_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError("refined_data 폴더에 CSV가 없습니다.")

    full_dfs: Dict[str, pd.DataFrame] = {}
    feature_cols_order: List[str] = []

    for csv_path in tqdm(csv_paths, desc="Reading & featurizing CSVs"):
        ticker = csv_path.stem
        tmp = pd.read_csv(csv_path)
        tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])

        df = add_features_and_labels(tmp, ticker)

        # 원본 컬럼 drop (DATE_COL은 유지해야 월별 필터링 가능)
        df = df.drop(columns=[OPEN_COL, CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL, "vol_ma_5", "vol_ma_20", "vol_ma_60"], errors="ignore")

        # clip
        # df = clip_features(df)

        full_dfs[ticker] = df

        # feature cols order
        cur_feats = [c for c in df.columns if c not in (LABEL_COL, DATE_COL)]
        if not feature_cols_order:
            feature_cols_order = cur_feats
        else:
            exist = set(feature_cols_order)
            for c in cur_feats:
                if c not in exist:
                    feature_cols_order.append(c)
                    exist.add(c)

    return full_dfs, feature_cols_order


def build_monthly_backtest_datasets():
    full_dfs, feature_cols = prepare_full_dfs()
    tickers = sorted(full_dfs.keys())
    print(f"Tickers loaded: {len(tickers)}")
    print(f"Num features: {len(feature_cols)}")
    all_cols = feature_cols + [LABEL_COL]

    # 월 시작 리스트 (MS = month start)
    month_starts = pd.date_range(TEST_START, TEST_END, freq="MS")
    three_years_before = month_starts - pd.DateOffset(years=3)
    # last_tag = month_starts[-1].strftime("%Y-%m-%d")
    last_tag = "2026-01-01"

    for test_start, train_start in zip(month_starts, three_years_before):
        test_end = (test_start + pd.offsets.MonthEnd(1))
        if test_end > TEST_END:
            test_end = TEST_END

        tag = test_start.strftime("%Y-%m-%d")
        out_base = BACKTEST_ROOT / tag
        train_out = out_base / "train"
        val_out   = out_base / "val"
        test_out  = out_base / "test"
        train_out.mkdir(parents=True, exist_ok=True)
        val_out.mkdir(parents=True, exist_ok=True)
        test_out.mkdir(parents=True, exist_ok=True)

        # --- split별 df 담기 ---
        train_dfs: Dict[str, pd.DataFrame] = {}
        val_dfs: Dict[str, pd.DataFrame] = {}
        test_dfs: Dict[str, pd.DataFrame] = {}

        # 1) test_start 이전 = train/val pool, test_start~test_end = test
        for tkr in tickers:
            df = full_dfs[tkr]

            # 날짜 필터링
            pool = df[(df[DATE_COL] >= train_start) & (df[DATE_COL] < test_start)].copy()
            # pool = df[df[DATE_COL] < test_start].copy()
            test = df[(df[DATE_COL] >= test_start) & (df[DATE_COL] <= test_end)].copy()

            pool = pool.sort_values(DATE_COL).reset_index(drop=True)
            test = test.sort_values(DATE_COL).reset_index(drop=True)

            pool_len = len(pool)
            test_len = len(test)

            tr, va = split_11_1(pool)

            train_dfs[tkr] = tr
            val_dfs[tkr] = va
            test_dfs[tkr] = test

        # 4) 표준화 통계는 "train만"으로 계산 (DATE_COL 제외)
        train_all = pd.concat(list(train_dfs.values()), ignore_index=True)

        feat_mean = train_all[feature_cols].mean()
        feat_std = train_all[feature_cols].std(ddof=0).replace(0.0, 1.0)

        # stats 저장 (각 월 폴더에)
        stats_path = out_base / "feature_standardize_stats.npz"
        np.savez(
            stats_path,
            all_cols=np.array(all_cols, dtype=object),
            mean=feat_mean.values.astype(np.float32),
            std=feat_std.values.astype(np.float32),
        )

        # 5) 정규화 + 텐서 저장 (train/val/test)
        for tkr in tickers:
            # DATE_COL 제거하고 standardize
            tr = train_dfs[tkr].drop(columns=[DATE_COL], errors="ignore")
            va = val_dfs[tkr].drop(columns=[DATE_COL], errors="ignore")
            te = test_dfs[tkr].drop(columns=[DATE_COL], errors="ignore")

            tr_std = standardize_df(tr, feature_cols, feat_mean, feat_std)
            va_std = standardize_df(va, feature_cols, feat_mean, feat_std)
            te_std = standardize_df(te, feature_cols, feat_mean, feat_std)

            # clip (표준화 후에도 동일 규칙 적용)
            tr_std = clip_features(tr_std)
            va_std = clip_features(va_std)
            te_std = clip_features(te_std)

            # label clip 일관 적용
            tr_std[LABEL_COL] = np.clip(tr_std[LABEL_COL].values, -10.0, 10.0)
            va_std[LABEL_COL] = np.clip(va_std[LABEL_COL].values, -10.0, 10.0)
            te_std[LABEL_COL] = np.clip(te_std[LABEL_COL].values, -10.0, 10.0)

            save_tensors(tr_std, feature_cols, train_out, tkr)
            save_tensors(va_std, feature_cols, val_out, tkr)
            save_tensors(te_std, feature_cols, test_out, tkr)

        print(f"[OK] {tag} saved -> {out_base} | pool_len={pool_len}, test_len={test_len}")

        # ===============================
        # 📊 마지막 반복 → 연도별 분포 Boxplot 저장
        # ===============================
        if tag == last_tag:
            print("Generating yearly distribution boxplots...")

            plot_dir = out_base / "yearly_boxplots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # train/val/test 전체 concat (DATE_COL 필요)
            all_concat = []
            for tkr in tickers:
                tr_df = train_dfs[tkr].drop(columns=[DATE_COL], errors="ignore")
                va_df = val_dfs[tkr].drop(columns=[DATE_COL], errors="ignore")
                te_df = test_dfs[tkr].drop(columns=[DATE_COL], errors="ignore")

                tr_std = standardize_df(tr_df, feature_cols, feat_mean, feat_std)
                va_std = standardize_df(va_df, feature_cols, feat_mean, feat_std)
                te_std = standardize_df(te_df, feature_cols, feat_mean, feat_std)

                tr_std[DATE_COL] = train_dfs[tkr][DATE_COL]
                va_std[DATE_COL] = val_dfs[tkr][DATE_COL]
                te_std[DATE_COL] = test_dfs[tkr][DATE_COL]

                tmp = pd.concat([tr_std, va_std, te_std], axis=0)
                all_concat.append(tmp)

            all_df = pd.concat(all_concat, ignore_index=True)

            # 연도 컬럼 생성
            all_df["year"] = all_df[DATE_COL].dt.year

            years = sorted(all_df["year"].unique())

            # feature + label 각각 plot
            plot_cols = feature_cols + [LABEL_COL]

            for col in plot_cols:
                plt.figure(figsize=(10, 6))

                data_per_year = []
                for y in years:
                    vals = all_df.loc[all_df["year"] == y, col].dropna().values
                    data_per_year.append(vals)

                plt.boxplot(data_per_year, tick_labels=years, showfliers=False)
                plt.title(f"Yearly Distribution: {col}")
                plt.xlabel("Year")
                plt.ylabel(col)

                save_path = plot_dir / f"{col}.png"
                plt.savefig(save_path, dpi=120, bbox_inches="tight")
                plt.close()

            print(f"[OK] Boxplots saved → {plot_dir}")


if __name__ == "__main__":
    build_monthly_backtest_datasets()