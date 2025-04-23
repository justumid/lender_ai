import psycopg2
import json
from typing import List, Dict, Optional

# Database config
DB_CONFIG = {
    "dbname": "credit_scoring",
    "user": "postgres",
    "password": "java2006",
    "host": "localhost",
    "port": 5432
}

def parse_json_field(raw: str) -> List[Dict]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        else:
            return []
    except Exception as e:
        print(f"[⚠️ JSON ERROR] {e}")
        return []

def fetch_salary_records(pinfl: str) -> List[Dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT salary_data FROM salary_records WHERE pinfl = %s;", (pinfl,))
            rows = cur.fetchall()
            return [item for row in rows for item in parse_json_field(row[0])]
    finally:
        conn.close()

def fetch_credit_records(pinfl: str) -> List[Dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT credit_data FROM credit_records WHERE "pPinfl" = %s;', (pinfl,))
            rows = cur.fetchall()
            records = []
            for row in rows:
                parsed = parse_json_field(row[0])
                for p in parsed:
                    records.append({"credit_data": p})
            return records
    finally:
        conn.close()

def get_unique_pinfls(limit: int = 1000) -> List[str]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT pinfl FROM salary_records LIMIT %s;", (limit,))
            rows = cur.fetchall()
            return [row[0] for row in rows]
    finally:
        conn.close()

def get_applicant_by_pinfl(pinfl: str) -> Optional[Dict]:
    salary = fetch_salary_records(pinfl)
    credit = fetch_credit_records(pinfl)
    if not salary and not credit:
        return None
    return {
        "pinfl": pinfl,
        "salary_records": salary,
        "credit_records": credit
    }
