import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.deep_encoder import DeepCreditEncoder
from models.transformer_heads import RiskClassifierHead, LoanLimitRegressor, FraudDetectorHead
from models.fraud_vae import FraudVAE
from models.simclr_encoder import SimCLREncoder
from trainer import DeepTrainer
from models.utils import log_memory_usage  # ✅ added

# 🔧 Configuration
EPOCHS = 30
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "demo_data/synthetic_dataset.json"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_dataset(path=DATA_PATH):
    with open(path, "r") as f:
        raw = json.load(f)

    X = []
    for rec in raw:
        seq = rec.get("sequence_tensor", [])
        if not seq or len(seq) != 12:
            continue
        X.append(seq)

    return torch.tensor(X, dtype=torch.float32)

def train():
    log_memory_usage("🚀 Before Training Loop")  # ✅

    X = load_dataset()
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder = DeepCreditEncoder().to(DEVICE)
    vae = FraudVAE(input_dim=15, seq_len=12).to(DEVICE)
    simclr = SimCLREncoder(input_dim=15, seq_len=12).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(vae.parameters()) +
        list(simclr.parameters()),
        lr=1e-3
    )

    loss_funcs = {
        "mse": nn.MSELoss()
    }

    trainer = DeepTrainer(
        models={
            "encoder": encoder,
            "vae": vae,
            "simclr": simclr
        },
        optim=optimizer,
        loss_funcs=loss_funcs,
        device=DEVICE
    )

    for epoch in range(1, EPOCHS + 1):
        log_memory_usage(f"🧠 Epoch {epoch}/{EPOCHS}")  # ✅
        losses = trainer.train_epoch(loader)

        print(f"✅ Losses: {losses}")

        torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, f"encoder_epoch{epoch}.pt"))
        torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, f"vae_epoch{epoch}.pt"))
        torch.save(simclr.state_dict(), os.path.join(CHECKPOINT_DIR, f"simclr_epoch{epoch}.pt"))

    # Final save
    torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "encoder.pt"))
    torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, "vae.pt"))
    torch.save(simclr.state_dict(), os.path.join(CHECKPOINT_DIR, "simclr.pt"))

    log_memory_usage("✅ After Training Complete")  # ✅
    print("✅ Training complete. All models saved.")

if __name__ == "__main__":
    train()
