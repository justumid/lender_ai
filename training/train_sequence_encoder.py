# training/train_sequence_encoder.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.deep_feature_encoder import SequenceFeatureEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sequence_data(path="demo_data/synthetic_dataset.json", max_len=12):
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sequences = []
    for entry in data:
        salary_data = entry.get("salary_data", [])
        seq = [(x.get("salary", 0), x.get("tax", 0)) for x in salary_data][-max_len:]
        padded = seq + [(0.0, 0.0)] * (max_len - len(seq))
        sequences.append(padded)

    return torch.tensor(sequences, dtype=torch.float32)  # shape: (N, T, 2)


def add_noise(batch, std=0.02):
    return batch + torch.randn_like(batch) * std


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent contrastive loss: encourages positive pairs to be similar
    """
    z = torch.cat([z_i, z_j], dim=0)      # (2B, D)
    z = F.normalize(z, dim=1)

    sim_matrix = torch.matmul(z, z.T) / temperature  # (2B, 2B)
    batch_size = z_i.size(0)
    labels = torch.arange(batch_size).to(DEVICE)

    # Mask self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(DEVICE)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    positives = torch.cat([
        torch.diag(sim_matrix, batch_size),
        torch.diag(sim_matrix, -batch_size)
    ])

    nominator = torch.exp(positives)
    denominator = torch.exp(sim_matrix).sum(dim=1)
    loss = -torch.log(nominator / denominator)
    return loss.mean()


def train_contrastive_encoder():
    print("🧠 Training Sequence Encoder with NT-Xent Loss...")

    X = load_sequence_data()  # shape: (N, T, 2)
    loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)

    model = SequenceFeatureEncoder(input_dim=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            x_i = add_noise(batch)
            x_j = add_noise(batch)

            z_i = model(x_i)
            z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📊 Epoch {epoch+1}/10 | Contrastive Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sequence_encoder.pt")
    print("✅ Saved sequence_encoder.pt")


if __name__ == "__main__":
    train_contrastive_encoder()
