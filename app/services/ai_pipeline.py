import torch
import numpy as np
from typing import Dict
from models.deep_encoder import DeepCreditEncoder
from models.transformer_heads import RiskClassifierHead, LoanLimitRegressor, FraudDetectorHead
from models.simclr_encoder import SimCLREncoder
from models.fraud_vae import FraudVAE
from app.services.preprocessing import extract_sequences
from app.services.explanation import explain_attention  # 👈 add this at the top


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔧 Load models
encoder = DeepCreditEncoder().to(device).eval()
risk_head = RiskClassifierHead().to(device).eval()
limit_head = LoanLimitRegressor().to(device).eval()
fraud_head = FraudDetectorHead().to(device).eval()
vae = FraudVAE().to(device).eval()
simclr = SimCLREncoder().to(device).eval()

def fuse_scores(risk_score: float, fraud_score: float, anomaly: float) -> float:
    """
    Fuses multiple scores into a final decision score.
    Weight fraud/anomaly more heavily if risk score is borderline.
    """
    base = 0.65 * risk_score + 0.25 * (1 - fraud_score) + 0.10 * np.exp(-anomaly)
    return float(np.clip(base, 0.0, 1.0))

def compute_confidence(risk_score: float, anomaly: float) -> str:
    if risk_score > 0.85 and anomaly < 0.05:
        return "HIGH"
    elif risk_score > 0.65:
        return "MEDIUM"
    else:
        return "LOW"

def run_full_scoring_pipeline(applicant: Dict) -> Dict:
    with torch.no_grad():
        # Step 1: Extract & preprocess
        salary_seq, credit_seq = extract_sequences(applicant)
        salary_seq = salary_seq.unsqueeze(0).to(device)
        credit_seq = credit_seq.unsqueeze(0).to(device)
        full_seq = torch.cat([salary_seq, credit_seq], dim=2)  # [1, 12, 10]

        # Step 2: Deep Encoding
        embedding = encoder(salary_seq, credit_seq)

        # Step 3: Score predictions
        risk_score = risk_head(embedding).item()
        loan_limit = limit_head(embedding).item()
        fraud_score = fraud_head(embedding).item()

        # Step 4: VAE anomaly score
        vae_anomaly = vae.anomaly_score(full_seq).item()

        # Step 5: SimCLR embedding
        simclr_vec = simclr(full_seq).cpu().numpy()
        segment_id = "SEG_UNKNOWN"

        # Step 6: Fusion and confidence
        final_score = fuse_scores(risk_score, fraud_score, vae_anomaly)
        confidence = compute_confidence(risk_score, vae_anomaly)

        attention_scores = explain_attention(applicant)

        # Decision thresholds (tunable)
        approval = "APPROVED" if final_score > 0.7 and fraud_score < 0.3 else "REJECTED"
        fraud_alert = "HIGH" if fraud_score > 0.6 or vae_anomaly > 0.4 else "LOW"

        return {
            "pinfl": applicant.get("pinfl"),
            "approval_decision": approval,
            "final_score": round(final_score * 100, 2),
            "loan_limit": round(loan_limit),
            "fraud_score": round(fraud_score, 4),
            "vae_anomaly": round(vae_anomaly, 4),
            "confidence_level": confidence,
            "segment_id": segment_id,
            "fraud_alert": fraud_alert,
            "explanations": attention_scores  # 👈 now contains salary & credit attention
        }

