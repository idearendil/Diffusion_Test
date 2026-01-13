from pathlib import Path

def rename_all_extensions_to_csv(folder_path: str) -> None:
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"폴더가 존재하지 않거나 디렉토리가 아닙니다: {folder}")

    for p in folder.iterdir():
        if not p.is_file():
            continue  # 하위 폴더 등은 건너뜀

        if p.suffix.lower() == ".csv":
            continue  # 이미 csv면 스킵

        new_path = p.with_suffix(".csv")

        # 같은 이름의 .csv가 이미 있으면 덮어쓰지 않고 건너뜀
        if new_path.exists():
            print(f"[SKIP] 이미 존재: {new_path.name} (원본: {p.name})")
            continue

        p.rename(new_path)
        print(f"[OK] {p.name} -> {new_path.name}")

# 사용 예시
rename_all_extensions_to_csv("raw_data")