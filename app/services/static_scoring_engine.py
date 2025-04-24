import numpy as np
import json

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def compute_static_score(applicant: dict) -> dict:
    """
    Compute static score from Excel-derived rules.
    Returns: {"static_score": int, "details": {...}}
    """
    salary_records = applicant.get("salary_records", [])
    credit_records = applicant.get("credit_records", [])

    breakdown = {}

    # --- 1️⃣ Average Salary Score ---
    recent_salaries = [
        safe_float(r.get("salary", 0))
        for r in salary_records[-6:]
        if safe_float(r.get("salary", 0)) > 0
    ]
    avg_salary = np.mean(recent_salaries) if recent_salaries else 0

    if avg_salary >= 6_000_000:
        salary_score = 25
    elif avg_salary >= 4_000_000:
        salary_score = 20
    elif avg_salary >= 2_000_000:
        salary_score = 15
    elif avg_salary > 0:
        salary_score = 10
    else:
        salary_score = 0

    breakdown["salary_score"] = salary_score

    # --- 2️⃣ Work Experience Score ---
    years = [safe_int(r.get("year")) for r in salary_records if safe_int(r.get("year", 0)) > 0]
    if years:
        experience_years = max(years) - min(years) + 1
    else:
        experience_years = 0

    if experience_years >= 5:
        experience_score = 15
    elif experience_years >= 3:
        experience_score = 10
    elif experience_years >= 1:
        experience_score = 5
    else:
        experience_score = 0

    breakdown["experience_score"] = experience_score

    # --- 3️⃣ Credit Overdue Score ---
    overdue_count = 0
    for record in credit_records:
        raw = record.get("credit_data", record)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except:
                continue
        contracts = raw.get("report", {}).get("contingent_liabilities", {}).get("contingent_liability", [])
        if not isinstance(contracts, list):
            contracts = [contracts]

        overdue_count += sum(
            1 for c in contracts if safe_float(c.get("overdue_debt_sum", 0)) > 0
        )

    if overdue_count == 0:
        overdue_score = 25
    elif overdue_count <= 2:
        overdue_score = 10
    else:
        overdue_score = 0

    breakdown["overdue_score"] = overdue_score

    # --- 4️⃣ Active Credit Contracts ---
    active_count = 0
    for record in credit_records:
        raw = record.get("credit_data", record)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except:
                continue
        contracts = raw.get("report", {}).get("contingent_liabilities", {}).get("contingent_liability", [])
        if not isinstance(contracts, list):
            contracts = [contracts]

        active_count += sum(1 for c in contracts if c.get("contract_status") == "1")

    if active_count <= 1:
        active_score = 15
    elif active_count <= 3:
        active_score = 10
    else:
        active_score = 5

    breakdown["active_credit_score"] = active_score

    # --- ✅ Final Static Score ---
    static_score = salary_score + experience_score + overdue_score + active_score
    static_score = min(static_score, 100)

    return {
        "static_score": static_score,
        "details": breakdown
    }
