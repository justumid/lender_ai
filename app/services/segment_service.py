# app/services/segment_service.py

import torch
import numpy as np
import joblib
from models.simclr_encoder import SimCLREncoder

SEGMENT_FEATURES = ["salary_mean_6mo", "salary_growth_rate", "salary_volatility_6mo",
                    "salary_std_vs_mean", "salary_drop_ratio", "employer_switch_count",
                    "tax_avg_6mo", "normalized_score", "credit_requests", "contracts_open"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = SimCLREncoder(input_dim=len(SEGMENT_FEATURES), projection_dim=64).to(DEVICE)
encoder.load_state_dict(torch.load("models/simclr_encoder.pt", map_location=DEVICE))
encoder.eval()

cluster_model = joblib.load("models/kmeans_segmenter.joblib")


def compute_user_segment(flat_features: dict) -> str:
    try:
        x = np.array([[flat_features[k] for k in SEGMENT_FEATURES]], dtype=np.float32)
        z = encoder(torch.tensor(x).to(DEVICE)).cpu().detach().numpy()
        label = cluster_model.predict(z)[0]
        return f"SEGMENT_{label}"
    except Exception as e:
        print(f"❌ Segment error: {e}")
        return "UNKNOWN"
