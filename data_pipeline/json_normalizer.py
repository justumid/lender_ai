from typing import List, Dict
from pydantic import BaseModel, validator


# ============================
# 📦 Pydantic Record Schemas
# ============================

class SalaryRecord(BaseModel):
    salary: float = 0.0
    tax: float = 0.0
    employer_changed: bool = False
    year: int = 2023
    period: int = 1

    @validator("period")
    def month_range(cls, v):
        return max(1, min(v, 12))  # clamp to 1–12


class CreditRecord(BaseModel):
    monthly_payment: float = 0.0
    remaining_term: int = 0
    overdue_days: int = 0
    is_open: bool = True
    contract_type: int = 0


class ApplicantNormalized(BaseModel):
    pinfl: str
    salary_records: List[SalaryRecord]
    credit_records: List[CreditRecord]


# ====================================
# 🚀 Normalization Interface
# ====================================

def normalize_applicant_json(raw_applicant: Dict) -> Dict:
    """
    Normalize a single applicant's record with validation.
    """
    return ApplicantNormalized(
        pinfl=raw_applicant.get("pinfl", ""),
        salary_records=raw_applicant.get("salary_records", []),
        credit_records=raw_applicant.get("credit_records", [])
    ).dict()


def normalize_batch(raw_applicants: List[Dict]) -> List[Dict]:
    """
    Normalize a list of applicants.
    Filters out invalid records.
    """
    cleaned = []
    for item in raw_applicants:
        try:
            normalized = normalize_applicant_json(item)
            cleaned.append(normalized)
        except Exception as e:
            print(f"[!] Skipping applicant due to validation error: {e}")
    return cleaned
