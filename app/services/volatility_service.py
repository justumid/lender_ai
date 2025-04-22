# app/services/volatility_service.py

from data_pipeline.feature_augment import extract_salary_sequence
import numpy as np


def compute_volatility_profile(applicant: dict) -> dict:
    salary_seq = extract_salary_sequence(applicant.get("salary_records", []))
    last6 = salary_seq[-6:] if salary_seq else []

    return {
        "volatility_std_6mo": float(np.std(last6)) if last6 else 0.0,
        "volatility_ratio": float(np.std(last6) / (np.mean(last6) + 1e-6)) if last6 else 0.0,
        "drop_events_6mo": int(sum(1 for i in range(1, len(last6)) if last6[i] < last6[i - 1])),
        "shock_flag": 1 if any(s < 0.5 * last6[i - 1] for i, s in enumerate(last6[1:], 1)) else 0
    }
