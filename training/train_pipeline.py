import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, r2_score

from models.deep_encoder import DeepCreditEncoder
from models.transformer_heads import RiskClassifierHead, LoanLimitRegressor, FraudDetectorHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(json_path="demo_data/synthetic_dataset.json"):
    import json
    from app.services.feature_generator import generate_sequences

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    X_seq, y_risk, y_limit, y_fraud = [], [], [], []

    for sample in records:
        seq = generate_sequences(sample)
        salary = seq["salary_sequence"]
        tax = seq["tax_sequence"]
        paired = list(zip(salary, tax))
        padded = paired[:24] + [(0, 0)] * max(0, 24 - len(paired))
        X_seq.append(padded)

        flat = sample["features"] if "features" in sample else sample  # fallback
        y_risk.append(1 if flat.get("normalized_score", 0) < 60 else 0)
        y_limit.append(float(flat.get("salary_mean_6mo", 0)) * 6)
        y_fraud.append(1 if flat.get("salary_volatility_6mo", 0) > 0.4 else 0)

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_risk, dtype=np.int64),
        np.array(y_limit, dtype=np.float32),
        np.array(y_fraud, dtype=np.int64)
    )


def train_model():
    X, y_risk, y_limit, y_fraud = load_dataset()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y_risk), torch.tensor(y_limit), torch.tensor(y_fraud))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder = DeepCreditEncoder(input_dim=2).to(DEVICE)
    risk_head = RiskClassifierHead().to(DEVICE)
    limit_head = LoanLimitRegressor().to(DEVICE)
    fraud_head = FraudDetectorHead().to(DEVICE)

    optim = torch.optim.Adam(
        list(encoder.parameters()) +
        list(risk_head.parameters()) +
        list(limit_head.parameters()) +
        list(fraud_head.parameters()),
        lr=1e-3
    )

    bce = torch.nn.BCELoss()
    mse = torch.nn.MSELoss()

    for epoch in range(10):
        encoder.train()
        total_loss = 0.0

        for batch_x, batch_risk, batch_limit, batch_fraud in loader:
            batch_x = batch_x.to(DEVICE)
            batch_risk = batch_risk.float().to(DEVICE)
            batch_limit = batch_limit.to(DEVICE)
            batch_fraud = batch_fraud.float().to(DEVICE)

            z = encoder(batch_x)
            pred_risk = risk_head(z)
            pred_limit = limit_head(z)
            pred_fraud = fraud_head(z)

            loss = (
                bce(pred_risk, batch_risk) +
                mse(pred_limit, batch_limit) +
                bce(pred_fraud, batch_fraud)
            )

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print(f"📊 Epoch {epoch+1} - Loss: {total_loss:.4f}")

    torch.save(encoder.state_dict(), "models/deep_encoder.pt")
    torch.save(risk_head.state_dict(), "models/risk_head.pt")
    torch.save(limit_head.state_dict(), "models/limit_head.pt")
    torch.save(fraud_head.state_dict(), "models/fraud_head.pt")
    print("✅ Models saved.")


if __name__ == "__main__":
    train_model()
