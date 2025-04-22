import psycopg2
import json
from typing import Dict, Optional
from data_pipeline.json_normalizer import normalize_applicant_json

# 🔧 Database configuration
DB_CONFIG = {
    "dbname": "credit_scoring",
    "user": "postgres",
    "password": "java2006",
    "host": "localhost",
    "port": 5432
}

def parse_json_field(raw):
    """
    Safely parse a raw JSON string or object from the database.
    Returns a list of dicts.
    """
    try:
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw

        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        else:
            return []
    except Exception as e:
        print(f"⚠️ JSON parsing error: {e}")
        return []


def fetch_salary_data(pinfl: str) -> Optional[dict]:
    """
    Fetch salary data for the given PINFL from the database.
    """
    query = "SELECT salary_data FROM salary_records WHERE pinfl = %s ORDER BY created_at DESC LIMIT 1;"
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(query, (pinfl,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def fetch_credit_data(pinfl: str) -> Optional[dict]:
    """
    Fetch credit data for the given PINFL from the database.
    """
    query = "SELECT credit_data FROM credit_records WHERE pPinfl = %s ORDER BY created_at DESC LIMIT 1;"
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(query, (pinfl,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_applicant_data(pinfl: str) -> dict:
    """
    Fetch salary and credit data for the given PINFL and return
    in structured format to be used by the scoring pipeline.
    """
    salary_data = fetch_salary_data(pinfl)
    credit_data = fetch_credit_data(pinfl)

    # Debugging: log the data fetching status
    if not salary_data:
        print(f"[DEBUG] No salary data for PINFL {pinfl}")
    if not credit_data:
        print(f"[DEBUG] No credit data for PINFL {pinfl}")

    # Ensure we don't raise error if one data type is missing
    return {
        "pinfl": pinfl,
        "salary_records": salary_data or [],
        "credit_records": credit_data or []
    }


def get_applicant_by_pinfl(pinfl: str) -> Optional[dict]:
    """
    Fetch salary and credit records for a given PINFL and return
    in structured format to be used by the scoring pipeline.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        # Fetch salary data
        cur.execute('SELECT salary_data FROM salary_records WHERE "pinfl" = %s;', (pinfl,))
        salary_rows = cur.fetchall()
        salary_records = []
        for row in salary_rows:
            salary_records.extend(parse_json_field(row[0]))

        # Fetch credit data
        cur.execute('SELECT "pClaimId", credit_data FROM credit_records WHERE "pPinfl" = %s;', (pinfl,))
        credit_rows = cur.fetchall()
        credit_records = []
        for claim_id, credit_data in credit_rows:
            for entry in parse_json_field(credit_data):
                credit_records.append({
                    "pPinfl": pinfl,
                    "pClaimId": claim_id,
                    "credit_data": entry
                })

        if not salary_records and not credit_records:
            return None  # Nothing found

        return {
            "pinfl": pinfl,
            "salary_records": salary_records,
            "credit_records": credit_records
        }

    finally:
        conn.close()


def build_dataset():
    """
    Build the full training dataset for the scoring pipeline.
    """
    pinfls = get_unique_pinfls()  # Ensure this returns a list of valid PINFLs
    dataset = []

    for pinfl in pinfls:
        try:
            applicant = get_applicant_by_pinfl(pinfl)
            if not applicant:
                print(f"[SKIP] No data for {pinfl}")
                continue

            # Process applicant data (normalize, extract sequences)
            salary_tensor, credit_tensor = extract_sequences(applicant)  # [12, 5], [12, 5]
            full_seq = torch.cat([salary_tensor, credit_tensor], dim=1)  # [12, 10]

            record = {
                "pinfl": pinfl,
                "sequence_tensor": full_seq.tolist(),
                "y_risk": 1,  # 🔧 Synthetic label — replace with true label if available
                "loan_limit": 25000000  # 🔧 Mock value — can replace with realistic calculation
            }

            dataset.append(record)

        except Exception as e:
            print(f"[SKIP] Failed on {pinfl}: {e}")
            continue

    # Save dataset
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[✅] Saved {len(dataset)} applicant records to {OUTPUT_PATH}")
