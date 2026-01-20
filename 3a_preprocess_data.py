import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# === 경로 & 컬럼 설정 ===
REFINED_DIR = Path("refined_data")
tensor_dir = Path("tensor_data")
tensor_dir.mkdir(exist_ok=True)
(tensor_dir / "train").mkdir(parents=True, exist_ok=True)
(tensor_dir / "val").mkdir(parents=True, exist_ok=True)
(tensor_dir / "test").mkdir(parents=True, exist_ok=True)

DATE_COL = "날짜"
OPEN_COL = "시가"
HIGH_COL = "고가"
LOW_COL = "저가"
CLOSE_COL = "종가"
VOLUME_COL = "거래량"

LABEL_COL = "label1"

# 등락률 컬럼이 refined_data에 없으므로, 종가로부터 만든 "일간 수익률(%)" 컬럼 이름
CHANGE_RATE_COL = "ret_pct"  # (종가 기준 % 수익률)


def save_feature_histograms_before_after(
    train_all: pd.DataFrame,
    feature_cols: List[str],
    mean: pd.Series,
    std: pd.Series,
    out_dir: Path,
    bins: int = 120,
    max_cols: int | None = None,   # 디버깅용: 일부 컬럼만 그리려면 숫자 넣기
) -> None:
    """
    train_all의 각 feature 컬럼에 대해
    - (좌) 정규화 전 히스토그램
    - (우) 정규화 후 히스토그램
    을 한 장으로 저장: out_dir/{col}.png

    주의: 컬럼명이 파일명으로 안전하지 않을 수 있어 약간 sanitize 함.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = feature_cols if max_cols is None else feature_cols[:max_cols]

    for col in tqdm(cols, desc="Saving feature histograms"):
        x_before = train_all[col].to_numpy(dtype=np.float64)

        # 정규화 후 값
        mu = float(mean[col])
        sd = float(std[col]) if float(std[col]) != 0.0 else 1.0
        x_after = (x_before - mu) / sd

        x_after = np.clip(x_after, -10.0, 10.0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(x_before, bins=bins)
        axes[0].set_title(f"{col} (before)")
        axes[0].set_xlabel("value")
        axes[0].set_ylabel("count")

        axes[1].hist(x_after, bins=bins)
        axes[1].set_title(f"{col} (after std)")
        axes[1].set_xlabel("value")
        axes[1].set_ylabel("count")

        fig.suptitle(col)
        fig.tight_layout()

        # 파일명 안전 처리 (슬래시/콜론 등 제거)
        safe_col = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in col)
        fig.savefig(out_dir / f"{safe_col}.png", dpi=50)
        plt.close(fig)

def is_volume_related(col: str) -> bool:
    # "거래량 제외"를 vol_ 파생 컬럼 전체로 해석
    # (원본 '거래량'은 drop되지만 vol_* 은 남아있음)
    return col.startswith("vol_") or ("거래량" in col)


def split_4_1_1(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """앞에서부터 순서대로 4:1:1 비율 split (train/val/test)."""
    n = len(df)
    train_end = (n * 4) // 6
    val_end = train_end + (n * 1) // 6

    # 안전장치
    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n - 1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


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
    df["high_low_spread"] = (df[HIGH_COL] - df[LOW_COL]) / (df[CLOSE_COL] + 1.0)

    # refined_data에는 등락률이 없으니 종가 기반으로 생성(%)  -> 원래의 CHANGE_RATE_COL 대체
    df[CHANGE_RATE_COL] = (df[CLOSE_COL] / (df[CLOSE_COL].shift(1) + 1e-12) - 1.0) * 100.0

    # 1) 거래량 = 거래량 * 시가 (원 코드 유지)
    # df[VOLUME_COL] = df[VOLUME_COL] * df[OPEN_COL]
    df["vol_ma_5"] = df[VOLUME_COL].rolling(window=5, min_periods=3).mean()
    df["vol_ma_20"] = df[VOLUME_COL].rolling(window=20, min_periods=6).mean()
    df["vol_ma_60"] = df[VOLUME_COL].rolling(window=60, min_periods=20).mean()
    df["vol_ratio_5"] = df[VOLUME_COL] / (df["vol_ma_5"] + 1.0)
    df["vol_ratio_20"] = df[VOLUME_COL] / (df["vol_ma_20"] + 1.0)
    df["vol_ratio_60"] = df[VOLUME_COL] / (df["vol_ma_60"] + 1.0)
    df["vol_diff"] = df[VOLUME_COL] / (df[VOLUME_COL].shift(1) + 1.0)
    df["vol_diff"] = np.clip(df["vol_diff"].values, 0.0, 10.0)
    df["vol_diff_ma_5"] = df["vol_diff"].diff().rolling(window=5, min_periods=3).mean()
    df["vol_diff_ma_20"] = df["vol_diff"].diff().rolling(window=20, min_periods=6).mean()
    df["vol_diff_ma_60"] = df["vol_diff"].diff().rolling(window=60, min_periods=20).mean()

    # 8) 과거 수익률 기반 평균/변동성
    for w in [5, 10, 20, 60]:
        minp = max(3, w // 3)
        df[f"ret_mean_{w}"] = df[CHANGE_RATE_COL].rolling(window=w, min_periods=minp).mean()
        df[f"ret_std_{w}"] = df[CHANGE_RATE_COL].rolling(window=w, min_periods=minp).std()

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
        df[f"close_to_max_{w}"] = df[CLOSE_COL] / (roll_max + 1.0)
        df[f"close_to_min_{w}"] = df[CLOSE_COL] / (roll_min + 1.0)

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
        df[f"atr_ratio_{w}"] = df[f"atr_{w}"] / (df[CLOSE_COL] + 1.0)

    # Candle Body Ratio / Upper/Lower Wick Ratio
    body = (df[CLOSE_COL] - df[OPEN_COL]).abs()
    upper_wick = df[HIGH_COL] - np.maximum(df[CLOSE_COL], df[OPEN_COL])
    lower_wick = np.minimum(df[CLOSE_COL], df[OPEN_COL]) - df[LOW_COL]
    range_ = (df[HIGH_COL] - df[LOW_COL]) + 1e-6
    df["candle_body_ratio"] = body / range_
    df["upper_wick_ratio"] = upper_wick / range_
    df["lower_wick_ratio"] = lower_wick / range_
    df["bullish"] = (df[CLOSE_COL] > df[OPEN_COL]).astype(np.float32)

    # ===== Label =====
    df[LABEL_COL] = (df[CLOSE_COL].shift(-1) / df[CLOSE_COL] - 1) * 10.0
    df[LABEL_COL] = (df[LABEL_COL] - np.mean(df[LABEL_COL])) / np.std(df[LABEL_COL])

    # 다음날 없는 마지막 행 + 이전 행들이 적은 첫 부분 행들 제거
    df = df.dropna()

    return df


def clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """split 전에: (label, vol_* 제외) feature를 -10~10 클리핑"""
    df = df.copy()
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    feature_cols = [c for c in feature_cols if c != "day_of_week" and c != "day_of_month"]
    clip_cols = [c for c in feature_cols if not is_volume_related(c)]
    if clip_cols:
        df[clip_cols] = df[clip_cols].clip(-10.0, 10.0)
    return df


def standardize_df(df: pd.DataFrame, feature_cols: List[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    df = df.copy()

    # 컬럼 순서 맞춤
    df = df[[*feature_cols, LABEL_COL]]

    # 표준화
    x = (df[feature_cols] - mean) / std
    x = x.fillna(0.0)  # 남은 NaN은 평균값(표준화 후 0)로 대체

    out = pd.concat([x, df[[LABEL_COL]].astype(np.float32)], axis=1)
    return out


def save_tensors(df: pd.DataFrame, feature_cols: List[str], out_dir: Path, ticker: str) -> None:
    x = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    torch.save(x_t, out_dir / f"{ticker}_x.pt")
    torch.save(y_t, out_dir / f"{ticker}_y.pt")


def build_dataset():
    # ticker별 (train/val/test) 저장
    train_dfs: Dict[str, pd.DataFrame] = {}
    val_dfs: Dict[str, pd.DataFrame] = {}
    test_dfs: Dict[str, pd.DataFrame] = {}

    # 길이 체크용
    full_lengths: Dict[str, int] = {}

    # feature 컬럼의 "전역 순서"를 보존하기 위해 첫 DF 기준으로 시작 + 이후 없는 컬럼 있으면 뒤에 추가
    feature_cols_order: List[str] = []

    csv_paths = sorted(REFINED_DIR.glob("*.csv"))
    if not csv_paths:
        print("refined_data 폴더에 CSV가 없습니다.")
        return

    for csv_path in tqdm(csv_paths, desc="Reading CSVs"):
        ticker = csv_path.stem
        tmp = pd.read_csv(csv_path)
        tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])

        df = add_features_and_labels(tmp, ticker)

        # 원본 컬럼 drop
        df = df.drop(columns=[DATE_COL, OPEN_COL, CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL, "vol_ma_5", "vol_ma_20", "vol_ma_60"], errors="ignore")

        # (요청) all_dfs에 넣기 전에: label, 거래량 관련 제외하고 -10~10 클리핑
        df = clip_features(df)

        # 길이 기록 + 동일 길이 체크는 루프 끝에서 한번에
        full_lengths[ticker] = len(df)

        # feature 컬럼 전역 순서 정하기(처음 본 컬럼 순서 유지)
        current_features = [c for c in df.columns if c != LABEL_COL]
        if not feature_cols_order:
            feature_cols_order = current_features
        else:
            # 기존에 없던 새 컬럼이 생기면 뒤에 추가
            existing = set(feature_cols_order)
            for c in current_features:
                if c not in existing:
                    feature_cols_order.append(c)
                    existing.add(c)

        # (요청) 4:1:1 순차 split
        tr, va, te = split_4_1_1(df)
        train_dfs[ticker] = tr
        val_dfs[ticker] = va
        test_dfs[ticker] = te

    print(feature_cols_order)
    if not train_dfs:
        print("유효한 데이터프레임이 없습니다.")
        return

    # ===== all_dfs 길이 동일 체크 =====
    lengths = list(full_lengths.values())
    if len(set(lengths)) != 1:
        # 어떤 ticker가 다른지 출력하고 중단
        base = max(set(lengths), key=lengths.count)
        bad = {t: l for t, l in full_lengths.items() if l != base}
        raise ValueError(f"모든 ticker DF 길이가 동일하지 않습니다. 기준길이={base}, 다른 ticker들={bad}")

    print(f"모든 ticker DF 길이 동일 확인: length={lengths[0]}")

    # ===== train만 합쳐서 mean/std 계산 =====
    train_all = pd.concat(list(train_dfs.values()), ignore_index=True)

    # label clip (원 코드 유지)
    train_all[LABEL_COL] = np.clip(train_all[LABEL_COL].values, -10.0, 10.0)

    # 통계 계산용 feature cols
    feature_cols_all = feature_cols_order[:]  # 순서 유지

    # std=0 방지
    feat_mean = train_all[feature_cols_all].mean()
    feat_std = train_all[feature_cols_all].std(ddof=0).replace(0.0, 1.0)

    # ===== (추가) feature별 분포 비교 히스토그램 저장 =====
    save_feature_histograms_before_after(
        train_all=train_all,
        feature_cols=feature_cols_all,
        mean=feat_mean,
        std=feat_std,
        out_dir=tensor_dir,   # tensor_data/{col}.png 로 저장
        bins=120
    )

    # stats 저장(추후 inference에 필요)
    stats_path = tensor_dir / "feature_standardize_stats.npz"
    np.savez(
        stats_path,
        feature_cols=np.array(feature_cols_all, dtype=object),
        mean=feat_mean.values.astype(np.float32),
        std=feat_std.values.astype(np.float32),
    )
    print(f"표준화 통계 저장: {stats_path}")

    # ===== label 분포 히스토그램(Train 기준) =====
    plt.figure(figsize=(6, 5))
    plt.hist(train_all[LABEL_COL].values, bins=100, alpha=0.7, edgecolor="black")
    plt.title("Distribution of label1 (train)")
    plt.xlabel("label1")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(tensor_dir / "label1_distribution_train.png")
    plt.close()
    print("라벨 분포 히스토그램 저장: label1_distribution_train.png")

    # ===== 정규화 + split별 tensor 저장 =====
    for ticker in tqdm(train_dfs.keys(), desc="Standardize & Save tensors"):
        # 각각 label clip 일관 적용
        train_dfs[ticker][LABEL_COL] = np.clip(train_dfs[ticker][LABEL_COL].values, -10.0, 10.0)
        val_dfs[ticker][LABEL_COL] = np.clip(val_dfs[ticker][LABEL_COL].values, -10.0, 10.0)
        test_dfs[ticker][LABEL_COL] = np.clip(test_dfs[ticker][LABEL_COL].values, -10.0, 10.0)

        tr_std = standardize_df(train_dfs[ticker], feature_cols_all, feat_mean, feat_std)
        va_std = standardize_df(val_dfs[ticker], feature_cols_all, feat_mean, feat_std)
        te_std = standardize_df(test_dfs[ticker], feature_cols_all, feat_mean, feat_std)

        tr_std = clip_features(tr_std)
        va_std = clip_features(va_std)
        te_std = clip_features(te_std)

        save_tensors(tr_std, feature_cols_all, tensor_dir / "train", ticker)
        save_tensors(va_std, feature_cols_all, tensor_dir / "val", ticker)
        save_tensors(te_std, feature_cols_all, tensor_dir / "test", ticker)

    print("완료: train/val/test 텐서 저장이 모두 끝났습니다.")


if __name__ == "__main__":
    build_dataset()