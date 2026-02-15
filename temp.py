import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# =========================
# 설정
# =========================
SRC_DIR = Path("backtest/true_regression_runs/2025-01-01")      # 복사할 원본 폴더
DEST_ROOT = Path("backtest/regression_runs")    # 붙여넣을 상위 폴더

DEST_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# 2020-01-01 ~ 2025-12-01 (72개월)
# =========================
dates = pd.date_range("2020-01-01", "2025-12-01", freq="MS")

for d in tqdm(dates):
    dest_dir = DEST_ROOT / d.strftime("%Y-%m-%d")
    
    # 이미 존재하면 덮어쓰기 (Python 3.8+)
    shutil.copytree(SRC_DIR, dest_dir, dirs_exist_ok=True)

print("완료: 72개 폴더 생성됨")