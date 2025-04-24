import os
import json
import torch
import psycopg2
from typing import List
from app.services.preprocessing import extract_sequence_tensor
from app.services.data_ingestion import get_applicant_by_pinfl
from app.services.static_scoring_engine import compute_static_score

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
    Fetch distinct PINFLs from salary_records for dataset generation.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pinfl FROM salary_records LIMIT %s", (limit,))
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"[ERROR] Could not fetch PINFLs: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def build_dataset():
    """
    Extracts input sequences and static scores for training.
    Output JSON format:
    {
        "pinfl": "string",
        "sequence_tensor": [[...], ...],  # shape [12, 15]
        "static_score": float
    }
    """
    dataset = []
    pinfls = get_unique_pinfls()

    for pinfl in pinfls:
        try:
            print(f"[INFO] Processing PINFL: {pinfl}")
            applicant = get_applicant_by_pinfl(pinfl)

            if not applicant:
                print(f"[SKIP] No applicant found for {pinfl}")
                continue

            if not applicant.get("salary_records") or not applicant.get("credit_records"):
                print(f"[SKIP] Incomplete data for {pinfl}")
                continue

            sequence_tensor = extract_sequence_tensor(applicant)  # [12, 15]
            static_result = compute_static_score(applicant)
            static_score = static_result["static_score"]

            dataset.append({
                "pinfl": pinfl,
                "sequence_tensor": sequence_tensor.tolist(),
                "static_score": float(static_score)
            })

        except Exception as e:
            print(f"[ERROR] Failed to process {pinfl}: {e}")
            continue

    # 💾 Save dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Saved {len(dataset)} applicants to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_dataset()
