import numpy as np
import json
import os

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calculate Population Stability Index between two distributions."""
    def scale(x):
        return np.clip(x, np.percentile(x, 1), np.percentile(x, 99))

    expected = scale(expected)
    actual = scale(actual)

    breakpoints = np.linspace(0, 1, buckets + 1)
    quantiles = np.quantile(expected, breakpoints)

    expected_hist, _ = np.histogram(expected, bins=quantiles)
    actual_hist, _ = np.histogram(actual, bins=quantiles)

    expected_perc = expected_hist / np.sum(expected_hist)
    actual_perc = actual_hist / np.sum(actual_hist)

    psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6)))
    return psi


def run_drift_monitor(train_path="demo_data/synthetic_dataset.json", recent_path="demo_data/recent_input_log.json"):
    if not os.path.exists(train_path) or not os.path.exists(recent_path):
        print("[ERROR] Training or live data file missing.")
        return

    with open(train_path) as f:
        train_data = json.load(f)

    with open(recent_path) as f:
        recent_data = json.load(f)

    train_features = np.array([r["sequence_tensor"] for r in train_data]).reshape(len(train_data), -1)
    recent_features = np.array([r["sequence_tensor"] for r in recent_data]).reshape(len(recent_data), -1)

    psi = calculate_psi(train_features.flatten(), recent_features.flatten())
    print(f"[🔍 PSI] Population Stability Index: {psi:.4f}")

    if psi > 0.2:
        print("[⚠️  DRIFT DETECTED] Consider retraining the model.")
    else:
        print("[✅] No significant drift.")

if __name__ == "__main__":
    run_drift_monitor()
