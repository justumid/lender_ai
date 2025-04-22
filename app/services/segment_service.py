import torch
import joblib
import numpy as np
from models.simclr_encoder import SimCLREncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔧 Load SimCLR encoder
simclr = SimCLREncoder().to(device).eval()

# (Optional) Load pretrained KMeans model if exists
try:
    kmeans = joblib.load("models/kmeans_segmenter.joblib")
    use_kmeans = True
except Exception:
    print("[segment_service] KMeans not loaded, returning raw vector.")
    kmeans = None
    use_kmeans = False

def compute_user_segment(full_sequence_tensor: torch.Tensor) -> str:
    """
    Uses SimCLR encoder to extract latent vector from [1, 12, 10] tensor.
    If KMeans is available, returns cluster label. Else returns "EMBED_..." string.
    """
    with torch.no_grad():
        full_sequence_tensor = full_sequence_tensor.to(device)  # [1, 12, 10]
        vec = simclr(full_sequence_tensor).cpu().numpy()  # [1, 64]

        if use_kmeans:
            label = int(kmeans.predict(vec)[0])
            return f"SEGMENT_{label}"
        else:
            vec_str = np.round(vec[0], 2)
            return f"EMBED_{vec_str[:5].tolist()}"  # Preview vector

