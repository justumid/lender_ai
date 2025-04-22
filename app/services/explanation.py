import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from models.deep_encoder import DeepCreditEncoder
from app.services.preprocessing import extract_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoder with exposed attention
encoder = DeepCreditEncoder(return_attention=True).to(device).eval()

def explain_attention(applicant: Dict) -> Dict:
    """
    Returns attention scores for salary and credit sequences.
    """
    with torch.no_grad():
        salary_seq, credit_seq = extract_sequences(applicant)
        salary_seq = salary_seq.unsqueeze(0).to(device)
        credit_seq = credit_seq.unsqueeze(0).to(device)

        # Run forward with attention
        embedding, attn_s, attn_c = encoder(salary_seq, credit_seq)

        # Get attention from first head of last layer [B, N, T, T]
        salary_attn = attn_s[-1][0, 0].cpu().numpy().tolist()
        credit_attn = attn_c[-1][0, 0].cpu().numpy().tolist()

        return {
            "salary_attention": salary_attn,
            "credit_attention": credit_attn
        }
