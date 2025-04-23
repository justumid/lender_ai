import torch
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any

SEQUENCE_LENGTH = 12
SALARY_DIM = 5
CREDIT_DIM = 10
TOTAL_DIM = SALARY_DIM + CREDIT_DIM

scaler_salary = StandardScaler()
scaler_credit = StandardScaler()

def safe_float(value: Any) -> float:
    try:
        return float(value)
    except:
        return 0.0

def extract_sequence_tensor(applicant: Dict[str, Any]) -> torch.Tensor:
    salary_records = applicant.get("salary_records", [])
    credit_records = applicant.get("credit_records", [])

    salary_data = []
    for r in salary_records[-SEQUENCE_LENGTH:]:
        salary_data.append([
            safe_float(r.get("salary")),
            safe_float(r.get("salaryTaxSum")),
            safe_float(r.get("inpsSum")),
            safe_float(r.get("year", 2023)),
            safe_float(r.get("period", 1)),
        ])
    while len(salary_data) < SEQUENCE_LENGTH:
        salary_data.insert(0, [0.0] * SALARY_DIM)
    salary_np = np.nan_to_num(np.array(salary_data, dtype=np.float32))

    salary_tensor = torch.zeros(SEQUENCE_LENGTH, SALARY_DIM)
    salary_tensor[:, :3] = torch.tensor(scaler_salary.fit_transform(salary_np[:, :3]), dtype=torch.float32)
    salary_tensor[:, 3] = torch.tensor(salary_np[:, 3] / 2100.0, dtype=torch.float32)
    salary_tensor[:, 4] = torch.tensor(salary_np[:, 4] / 12.0, dtype=torch.float32)

    credit_data = []
    for r in credit_records:
        raw = r.get("credit_data", r)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except:
                continue
        entries = raw.get("report", {}).get("contingent_liabilities", {}).get("contingent_liability", [])
        if not isinstance(entries, list):
            entries = [entries]

        for c in entries:
            credit_data.append([
                safe_float(c.get("overdue_debt_sum")),
                safe_float(c.get("amount")),
                safe_float(c.get("percent")),
                safe_float(c.get("max_uninter_overdue_percent")),
                safe_float(c.get("total_debt_sum")),
                safe_float(c.get("security_amount")),
                safe_float(c.get("amount_issued")),
                safe_float(c.get("actual_monthly_average_payment")),
                safe_float(c.get("class_asset_quality")),
                1.0 if c.get("contract_status") == "1" else 0.0,
            ])

    while len(credit_data) < SEQUENCE_LENGTH:
        credit_data.insert(0, [0.0] * CREDIT_DIM)
    credit_np = np.nan_to_num(np.array(credit_data[-SEQUENCE_LENGTH:], dtype=np.float32))
    credit_tensor = torch.tensor(scaler_credit.fit_transform(credit_np), dtype=torch.float32)

    # Combine salary and credit: [12, 5] + [12, 10] → [12, 15]
    full_tensor = torch.cat([salary_tensor, credit_tensor], dim=1)
    return full_tensor
