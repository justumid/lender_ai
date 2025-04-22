import torch
from torch.utils.data import DataLoader, TensorDataset
from models.fraud_vae import FraudVAE
from models.loss import vae_loss
import numpy as np
import joblib

def train_vae(X: np.ndarray, model_path: str = "models/vae_fraud.pt"):
    model = FraudVAE(input_dim=X.shape[1]).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=64, shuffle=True)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for (batch,) in data_loader:
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - VAE Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_path)
