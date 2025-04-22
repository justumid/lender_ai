import torch
import numpy as np
import joblib
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from models.simclr_encoder import SimCLREncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_segmenter(X: np.ndarray, model_path="models/kmeans_segmenter.joblib", encoder_path="models/simclr_encoder.pt", input_dim=None):
    input_dim = input_dim or X.shape[1]

    # Load trained encoder
    model = SimCLREncoder(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    model.eval()

    # Generate embeddings
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=64)
    embeddings = []

    with torch.no_grad():
        for (batch,) in loader:
            z = model(batch.to(DEVICE)).cpu().numpy()
            embeddings.append(z)

    all_embeddings = np.vstack(embeddings)

    # Fit KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(all_embeddings)

    joblib.dump(kmeans, model_path)
    print(f"✅ KMeans segmenter saved to {model_path}")
