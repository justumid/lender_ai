# app/services/ai_pipeline.py

import numpy as np
import torch
from models.deep_encoder import DeepCreditEncoder
from models.transformer_heads import RiskClassifierHead, LoanLimitRegressor, FraudDetectorHead
from models.shap_explainer import ShapExplainer

from app.services.preprocessing import preprocess_features
from app.services.fraud_detector import compute_fraud_risk
from app.services.volatility_service import compute_volatility_profile
from app.services.segment_service import compute_user_segment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
encoder = DeepCreditEncoder(input_dim=22).to(DEVICE)
encoder.load_state_dict(torch.load("models/deep_encoder.pt", map_location=DEVICE))
encoder.eval()

risk_head = RiskClassifierHead().to(DEVICE)
risk_head.load_state_dict(torch.load("models/risk_head.pt", map_location=DEVICE))
risk_head.eval()

limit_head = LoanLimitRegressor().to(DEVICE)
limit_head.load_state_dict(torch.load("models/limit_head.pt", map_location=DEVICE))
limit_head.eval()

fraud_head = FraudDetectorHead().to(DEVICE)
fraud_head.load_state_dict(torch.load("models/fraud_head.pt", map_location=DEVICE))
fraud_head.eval()

shap_explainer = None


def run_full_ai_pipeline(applicant: dict) -> dict:
    pinfl = applicant.get("pinfl")

    # Step 1: Preprocess
    flat_features = preprocess_features(applicant)
    input_array = np.array([list(flat_features.values())[1:]], dtype=np.float32)
    input_tensor = torch.tensor(input_array).to(DEVICE)

    # Step 2: Encode embedding
    with torch.no_grad():
        embedding = encoder(input_tensor)

    # Step 3: Predict scores
    credit_score = float(risk_head(embedding).cpu().item()) * 100
    loan_limit = float(limit_head(embedding).cpu().item())
    fraud_risk = float(fraud_head(embedding).cpu().item())

    # Step 4: Explain
    global shap_explainer
    try:
        if shap_explainer is None:
            shap_explainer = ShapExplainer(risk_head, list(flat_features.keys())[1:])
        explanations = shap_explainer.explain(input_array)
    except Exception as e:
        print(f"⚠️ SHAP explanation failed: {e}")
        explanations = {k: 0.0 for k in list(flat_features.keys())[1:][:5]}

    # Step 5: Segment + volatility
    segment_id = compute_user_segment(flat_features)
    volatility_profile = compute_volatility_profile(applicant)

    return {
        "pinfl": pinfl,
        "credit_score": round(credit_score, 2),
        "decision": "APPROVED" if credit_score >= 60 else "REJECTED",
        "loan_limit": round(loan_limit, 2) if credit_score >= 60 else None,
        "fraud_risk": round(fraud_risk, 4),
        "segment_id": segment_id,
        "volatility_profile": volatility_profile,
        "explanations": explanations
    }
