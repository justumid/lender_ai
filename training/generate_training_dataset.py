import psycopg2
import json
from typing import List
from app.services.data_ingestion import get_applicant_by_pinfl
from app.services.preprocessing import extract_sequences

# 🔧 DB config — ensure it matches your actual DB
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
    Extract a list of unique PINFLs from salary_records table.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pinfl FROM salary_records LIMIT %s", (limit,))
        results = cursor.fetchall()
        return [row[0] for row in results]
    except Exception as e:
        print(f"[ERROR] Could not fetch pinfls: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def build_dataset():
    """
    Build the full training dataset for the scoring pipeline.
    """
    pinfls = get_unique_pinfls()  # Ensure this returns a list of valid PINFLs
    dataset = []

    for pinfl in pinfls:
        try:
            print(f"[DEBUG] Processing PINFL: {pinfl}")  # Debugging line
            applicant = get_applicant_by_pinfl(pinfl)
            
            if not applicant:
                print(f"[SKIP] No data found for PINFL {pinfl}")
                continue
            
            # Ensure salary_records and credit_records are both available
            if not applicant.get("salary_records") or not applicant.get("credit_records"):
                print(f"[SKIP] Incomplete data for PINFL {pinfl}")
                continue

            # Process the data: normalize, extract sequences, and concatenate
            salary_tensor, credit_tensor = extract_sequences(applicant)  # [12, 5], [12, 5]
            full_seq = torch.cat([salary_tensor, credit_tensor], dim=1)  # [12, 10]

            # Mock labels for training: 
            # Replace synthetic values with real data if available.
            record = {
                "pinfl": pinfl,
                "sequence_tensor": full_seq.tolist(),  # Convert tensor to list for saving
                "y_risk": 1,  # 🔧 Synthetic label — replace with true label if available
                "loan_limit": 25000000  # 🔧 Mock value — can replace with realistic calculation
            }

            dataset.append(record)

        except Exception as e:
            print(f"[SKIP] Failed on PINFL {pinfl}: {e}")
            continue

    # Save dataset to JSON
    output_path = "demo_data/synthetic_dataset.json"  # Output path for the dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[✅] Saved {len(dataset)} applicant records to {output_path}")


    # Save dataset
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[✅] Saved {len(dataset)} applicant records to {OUTPUT_PATH}")

if __name__ == "__main__":
    import torch
    build_dataset()
