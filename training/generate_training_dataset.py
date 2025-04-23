import os
import json
import torch
import psycopg2
from typing import List
from app.services.preprocessing import extract_sequence_tensor
from app.services.data_ingestion import get_applicant_by_pinfl

# 🔧 Database Configuration
DB_CONFIG = {
    "dbname": "credit_scoring",
    "user": "postgres",
    "password": "java2006",
    "host": "localhost",
    "port": 5432
}

OUTPUT_PATH = "demo_data/synthetic_dataset.json"

def get_unique_pinfls(limit=1000) -> List[str]:
    """
    Get unique PINFLs from the salary_records table.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pinfl FROM salary_records LIMIT %s", (limit,))
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        print(f"[ERROR] Failed to fetch PINFLs: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def build_dataset():
    """
    Construct unsupervised dataset by extracting only sequences [12, 15].
    """
    dataset = []
    pinfls = get_unique_pinfls()

    for pinfl in pinfls:
        try:
            print(f"[INFO] Processing {pinfl}")
            applicant = get_applicant_by_pinfl(pinfl)
            if not applicant:
                print(f"[SKIP] No records for {pinfl}")
                continue

            if not applicant.get("salary_records") or not applicant.get("credit_records"):
                print(f"[SKIP] Missing salary or credit data for {pinfl}")
                continue

            sequence_tensor = extract_sequence_tensor(applicant)  # [12, 15]

            dataset.append({
                "pinfl": pinfl,
                "sequence_tensor": sequence_tensor.tolist()
            })

        except Exception as e:
            print(f"[ERROR] PINFL {pinfl} failed: {e}")
            continue

    # 💾 Save to file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Saved {len(dataset)} records to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_dataset()
