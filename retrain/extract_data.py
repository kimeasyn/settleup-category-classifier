# retrain/extract_data.py
import psycopg2
import csv
import os

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USERNAME")
DB_PASS = os.getenv("DB_PASSWORD")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./retrain/output")

def extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

    # description이 있는 모든 로그 추출
    cur = conn.cursor()
    cur.execute("""
        SELECT description, final_category
        FROM prediction_logs
        WHERE description IS NOT NULL
          AND final_category IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 50:  # 50건 미만이면 스킵
        print(f"새 데이터 {len(rows)}건 < 50건. 재학습 스킵.")
        with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
            json.dump({"skip": True, "reason": "insufficient_data", "count": len(rows)}, f)
        sys.exit(0)

    # CSV 저장
    output_path = os.path.join(OUTPUT_DIR, "new_data.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "category"])
        writer.writerows(rows)

    print(f"Extracted {len(rows)} rows → {output_path}")
    existing_path = os.getenv("EXISTING_DATA", "./data/train.csv")
    merged_path = os.path.join(OUTPUT_DIR, "merged_train.csv")

    merged = []

    # 기존 데이터 로드
    if os.path.exists(existing_path):
        with open(existing_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merged.append((row["description"], row["category"]))
        print(f"Existing data: {len(merged)} rows")

    # 새 데이터 추가
    merged.extend(rows)
    print(f"Total merged: {len(merged)} rows")

    # 저장
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "category"])
        writer.writerows(merged)

    print(f"Saved → {merged_path}")

if __name__ == "__main__":
    extract()