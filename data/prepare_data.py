from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ── 프로젝트 루트 ──
DATA_DIR = Path(__file__).resolve().parent

# result.xlsx 읽기
df = pd.read_excel(DATA_DIR / "result.xlsx")
print(f"전체 데이터: {len(df)}건")
print(f"\n카테고리별 분포:")
print(df["category"].value_counts())

# 카테고리별 비율 유지하며 8:2 분리
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["category"]
)

train_df.to_csv(DATA_DIR / "train.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(DATA_DIR / "test.csv", index=False, encoding="utf-8-sig")

print(f"\ntrain.csv: {len(train_df)}건")
print(f"test.csv: {len(test_df)}건")
