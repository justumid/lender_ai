import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.simclr_encoder import SimCLREncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    z = torch.nn.functional.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(len(z), dtype=torch.bool).to(z.device)
    sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

    positives = torch.cat([torch.diag(sim_matrix, len(z_i)), torch.diag(sim_matrix, -len(z_i))])
    loss = -positives + torch.logsumexp(sim_matrix, dim=1)
    return loss.mean()


def train_simclr(X: np.ndarray, model_path="models/simclr_encoder.pt", input_dim=None):
    input_dim = input_dim or X.shape[1]

    model = SimCLREncoder(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=64, shuffle=True)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for (batch,) in dataset:
            batch = batch.to(DEVICE)
            x_i = batch + torch.randn_like(batch) * 0.01
            x_j = batch + torch.randn_like(batch) * 0.01

            z_i = model(x_i)
            z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📊 SimCLR Epoch {epoch+1}/10 | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ SimCLR encoder saved to {model_path}")
