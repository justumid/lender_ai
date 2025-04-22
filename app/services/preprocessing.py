# app/services/preprocessing.py

import numpy as np
from typing import Dict, Any
from data_pipeline.feature_augment import extract_salary_sequence, extract_tax_sequence, extract_employer_sequence
from data_pipeline.fraud_features import extract_credit_events, credit_instability_signals


def salary_growth_rate(seq):
    return (seq[-1] - seq[-2]) / (seq[-2] + 1e-6) if len(seq) >= 2 else 0.0

def salary_volatility(seq):
    return float(np.std(seq[-6:])) if len(seq) >= 6 else 0.0

def salary_std_vs_mean(seq):
    return float(np.std(seq) / (np.mean(seq) + 1e-6)) if seq else 0.0

def salary_drop_ratio(seq):
    return sum(1 for i in range(1, len(seq)) if seq[i] < seq[i-1]) / len(seq) if len(seq) > 1 else 0.0


def preprocess_features(applicant: Dict[str, Any]) -> Dict[str, float]:
    salary_records = applicant.get("salary_records", [])
    credit_records = applicant.get("credit_records", [])

    salary_seq = extract_salary_sequence(salary_records)
    tax_seq = extract_tax_sequence(salary_records)
    employer_seq = extract_employer_sequence(salary_records)

    credit_events = extract_credit_events(credit_records)
    credit_vector = credit_instability_signals(credit_events)

    features = {
        "salary_mean_6mo": round(np.mean(salary_seq[-6:]), 2) if salary_seq else 0.0,
        "salary_growth_rate": salary_growth_rate(salary_seq),
        "salary_volatility_6mo": salary_volatility(salary_seq),
        "salary_std_vs_mean": salary_std_vs_mean(salary_seq),
        "salary_drop_ratio": salary_drop_ratio(salary_seq),
        "employer_switch_count": sum(1 for i in range(1, len(employer_seq)) if employer_seq[i] != employer_seq[i-1]),
        "tax_avg_6mo": round(np.mean(tax_seq[-6:]), 2) if tax_seq else 0.0,
        **credit_vector
    }

    # Add static safety for numerical types
    for k, v in features.items():
        try:
            features[k] = float(v)
        except Exception:
            features[k] = 0.0

    return {"pinfl": applicant.get("pinfl"), **features}
