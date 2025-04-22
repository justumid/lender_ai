import pandas as pd
import numpy as np
from typing import Dict


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) for one feature.
    """
    def _get_percents(x):
        hist, _ = np.histogram(x, bins=buckets)
        return hist / len(x)

    expected_pct = _get_percents(expected)
    actual_pct = _get_percents(actual)

    psi = np.sum([
        (e - a) * np.log((e + 1e-6) / (a + 1e-6))
        for e, a in zip(expected_pct, actual_pct)
    ])
    return round(psi, 4)


def detect_feature_drift(baseline: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.2) -> Dict[str, float]:
    """
    Returns a dict of features with PSI > threshold.
    """
    drifted = {}
    common = list(set(baseline.columns) & set(current.columns))

    for col in common:
        try:
            psi = calculate_psi(baseline[col].dropna().values, current[col].dropna().values)
            if psi > threshold:
                drifted[col] = psi
        except Exception:
            continue

    return drifted
