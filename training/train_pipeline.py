import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import os

from models.deep_encoder import DeepCreditEncoder
from models.transformer_heads import RiskClassifierHead, LoanLimitRegressor, FraudDetectorHead
from models.fraud_vae import FraudVAE
from models.simclr_encoder import SimCLREncoder
from trainer import DeepTrainer

# 🔧 Training configuration
EPOCHS = 30
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "demo_data/synthetic_dataset.json"

def load_dataset(path=DATA_PATH):
    with open(path, "r") as f:
        raw_data = json.load(f)

    X, y_risk, y_limit = [], [], []
    for rec in raw_data:
        seq = rec.get("sequence_tensor", [])
        if not seq or len(seq) != 12:
            continue
        X.append(seq)
        y_risk.append(rec.get("y_risk", 0))
        y_limit.append(rec.get("loan_limit", 0))

    X = torch.tensor(X, dtype=torch.float32)
    y_risk = torch.tensor(y_risk, dtype=torch.float32)
    y_limit = torch.tensor(y_limit, dtype=torch.float32)
    return X, y_risk, y_limit

def train():
    # Load and prepare dataset
    X, y_risk, y_limit = load_dataset()
    dataset = TensorDataset(X, y_risk, y_limit)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models
    encoder = DeepCreditEncoder().to(DEVICE)
    risk_head = RiskClassifierHead().to(DEVICE)
    limit_head = LoanLimitRegressor().to(DEVICE)
    fraud_head = FraudDetectorHead().to(DEVICE)
    vae = FraudVAE().to(DEVICE)
    simclr = SimCLREncoder().to(DEVICE)

    # Optimizer and loss functions
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(risk_head.parameters()) +
        list(limit_head.parameters()) +
        list(fraud_head.parameters()) +
        list(vae.parameters()) +
        list(simclr.parameters()),
        lr=1e-3
    )

    loss_funcs = {
        "bce": nn.BCELoss(),
        "mse": nn.MSELoss()
    }

    # Trainer class
    trainer = DeepTrainer(
        models={
            "encoder": encoder,
            "risk": risk_head,
            "limit": limit_head,
            "fraud": fraud_head,
            "vae": vae,
            "simclr": simclr
        },
        optim=optimizer,
        loss_funcs=loss_funcs,
        device=DEVICE
    )

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        losses = trainer.train_epoch(loader)
        print("Losses:", losses)

    # Save models
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), "checkpoints/encoder.pt")
    torch.save(risk_head.state_dict(), "checkpoints/risk_head.pt")
    torch.save(limit_head.state_dict(), "checkpoints/limit_head.pt")
    torch.save(fraud_head.state_dict(), "checkpoints/fraud_head.pt")
    torch.save(vae.state_dict(), "checkpoints/vae.pt")
    torch.save(simclr.state_dict(), "checkpoints/simclr.pt")

if __name__ == "__main__":
    train()
