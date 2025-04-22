# app/services/fraud_detector.py

import numpy as np
import torch
from models.fraud_vae import FraudVAE
from models.transformer_heads import FraudDetectorHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
vae = FraudVAE(input_dim=22).to(DEVICE)
vae.load_state_dict(torch.load("models/fraud_vae.pt", map_location=DEVICE))
vae.eval()

fraud_head = FraudDetectorHead(input_dim=8).to(DEVICE)  # VAE bottleneck dim
fraud_head.load_state_dict(torch.load("models/fraud_head.pt", map_location=DEVICE))
fraud_head.eval()


def compute_fraud_risk(flat_features: dict) -> float:
    try:
        x = np.array([list(flat_features.values())[1:]], dtype=np.float32)
        with torch.no_grad():
            z = vae.encode(torch.tensor(x).to(DEVICE))
            risk = fraud_head(z).cpu().item()
        return round(float(risk), 4)
    except Exception as e:
        print(f"⚠️ Fraud detection failed: {e}")
        return 0.0
