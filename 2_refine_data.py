import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = "raw_data"
OUT_DIR = "refined_data"

REF_FILE = "005930.csv"

START_DATE = "2016-11-01"
END_DATE   = "2025-12-31"

DATE_COL = "날짜"
DROP_COLS = ["등락률"]

# 원본에 존재할 컬럼들
PRICE_COLS = ["시가", "고가", "저가", "종가"]
VOL_COL = "거래량"

# 보간/ffill/bfill을 수행할 컬럼 (요청사항)
INTERP_COLS = ["시가", "거래량"]


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_reference_calendar(raw_dir: str) -> pd.DatetimeIndex:
    ref_path = os.path.join(raw_dir, REF_FILE)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"기준 파일 '{ref_path}' 를 찾을 수 없습니다.")

    ref = pd.read_csv(ref_path)
    if DATE_COL not in ref.columns:
        raise ValueError(f"기준 파일 '{REF_FILE}'에 '{DATE_COL}' 컬럼이 없습니다.")

    ref[DATE_COL] = pd.to_datetime(ref[DATE_COL], errors="coerce")
    ref = ref.dropna(subset=[DATE_COL])

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    ref = ref[(ref[DATE_COL] >= start) & (ref[DATE_COL] <= end)].copy()

    cal = pd.DatetimeIndex(ref[DATE_COL].drop_duplicates().sort_values())
    if len(cal) == 0:
        raise ValueError(f"'{REF_FILE}'에서 {START_DATE}~{END_DATE} 구간 날짜가 비어 있습니다.")
    return cal


def preprocess_one_csv(in_path: str, out_path: str, cal_dates: pd.DatetimeIndex) -> None:
    df = pd.read_csv(in_path)

    if DATE_COL not in df.columns:
        raise ValueError(f"[{in_path}] '{DATE_COL}' 컬럼이 없습니다.")

    # 날짜 파싱/정렬
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    # 등락률 제거(있으면)
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 범위 필터
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    df = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)].copy()

    # index로 두고 기준 달력으로 reindex (005930 날짜들만)
    df = df.set_index(DATE_COL)

    # 필요한 컬럼들 존재 체크/생성 + numeric 변환
    for c in PRICE_COLS + [VOL_COL]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reindex(cal_dates)

    # 보간 필요 여부(= predictable==0의 근거) 판단은
    # "시가/거래량 중 하나라도 NaN" 또는 "시가==0" 이면 보간 필요로 잡는 게 자연스러움
    created_missing = df[INTERP_COLS].isna().any(axis=1)

    # 시가==0인 행: (날짜 제외) 시가/거래량을 NaN 처리 -> 보간 대상으로
    open_zero_mask = df["시가"].fillna(np.nan).eq(0)
    df.loc[open_zero_mask, INTERP_COLS] = np.nan

    interpolation_needed = created_missing | open_zero_mask

    # 1) 시가/거래량만 time 보간
    df[INTERP_COLS] = df[INTERP_COLS].interpolate(method="time")

    # 2) 양 끝 NaN은 시가/거래량만 bfill/ffill
    df[INTERP_COLS] = df[INTERP_COLS].bfill().ffill()

    # 3) 고가/저가/종가는 항상 시가와 동일하게 강제
    df["고가"] = df["고가"].fillna(df["시가"])
    df["저가"] = df["저가"].fillna(df["시가"])
    df["종가"] = df["종가"].fillna(df["시가"])

    # predictable: 보간이 필요했던 날은 0, 나머지 1
    predictable = (~interpolation_needed).astype(int)

    # "보간 구간의 바로 직전 행도 0"  (전날을 0으로)
    prev_day = interpolation_needed.shift(-1, fill_value=False)
    predictable[prev_day] = 0

    df["predictable"] = predictable.values

    # 등락률이 극단적인 행도 predictable=0
    df["predictable"] = np.where(
        (df["종가"] / df["시가"] >= 1.25) | (df["종가"] / df["시가"] <= 0.75),
        0,
        df.get("predictable", 1)  # 기존 값 유지, 없으면 기본 1
    )

    # 거래량이 너무 적은 행도 predictable=0
    df["predictable"] = np.where(
        (df["거래량"] * df["시가"] <= 2000000000),
        0,
        df.get("predictable", 1)  # 기존 값 유지, 없으면 기본 1
    )

    # 날짜 컬럼 복구
    df = df.reset_index().rename(columns={"index": DATE_COL})

    # 저장
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def main():
    ensure_out_dir(OUT_DIR)

    cal_dates = load_reference_calendar(RAW_DIR)
    print(f"Reference calendar loaded from {REF_FILE}: {len(cal_dates)} dates")

    csv_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"'{RAW_DIR}' 폴더에 CSV 파일이 없습니다.")

    for in_path in tqdm(csv_paths):
        filename = os.path.basename(in_path)
        out_path = os.path.join(OUT_DIR, filename)
        preprocess_one_csv(in_path, out_path, cal_dates)


if __name__ == "__main__":
    main()