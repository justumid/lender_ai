import os
import json
from models.utils import to_numpy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.deep_encoder import DeepCreditEncoder
from models.fraud_vae import FraudVAE
from models.simclr_encoder import SimCLREncoder
from training.trainer import DeepTrainer
from models.utils import log_memory_usage
from app.services.preprocessing import extract_sequence_tensor
from app.schemas.scoring_response import ScoringResponse
from models.transformer_heads import RiskClassifierHead, LoanLimitRegressor, FraudDetectorHead


# 🔧 Config
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "demo_data/synthetic_dataset.json"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_dataset(path=DATA_PATH):
    """
    Loads the [12, 15] sequence tensor dataset from JSON into a PyTorch Tensor.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    X = []
    for rec in raw:
        seq = rec.get("sequence_tensor", [])
        if not seq or len(seq) != 12 or len(seq[0]) != 15:
            continue
        X.append(seq)

    return torch.tensor(X, dtype=torch.float32)

def save_model(model, name: str, epoch: int = None):
    """
    Saves model weights to the checkpoint directory.
    """
    suffix = f"_epoch{epoch}" if epoch else ""
    path = os.path.join(CHECKPOINT_DIR, f"{name}{suffix}.pt")
    torch.save(model.state_dict(), path)

def build_models() -> dict:
    """
    Initializes and returns all models used in training.
    """
    return {
        "encoder": DeepCreditEncoder().to(DEVICE),
        "vae": FraudVAE(input_dim=15, seq_len=12).to(DEVICE),
        "simclr": SimCLREncoder(input_dim=15, seq_len=12).to(DEVICE),
        "risk_head": RiskClassifierHead(input_dim=128).to(DEVICE),
        "fraud_head": FraudDetectorHead(input_dim=64).to(DEVICE),
        "limit_head": LoanLimitRegressor(input_dim=128).to(DEVICE)
    }

def train():
    """
    Full training loop for encoder, VAE, SimCLR, and model heads (risk_head, fraud_head, limit_head).
    """
    log_memory_usage("🚀 Starting Training")

    # Load dataset
    X = load_dataset()
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models
    models = build_models()

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        list(models["encoder"].parameters()) +
        list(models["vae"].parameters()) +
        list(models["simclr"].parameters()) +
        list(models["risk_head"].parameters()) +
        list(models["fraud_head"].parameters()) +
        list(models["limit_head"].parameters()),
        lr=LEARNING_RATE
    )

    # Initialize trainer
    trainer = DeepTrainer(
        models=models,
        optim=optimizer,
        loss_funcs={"mse": nn.MSELoss()},
        device=DEVICE
    )

    for epoch in range(1, EPOCHS + 1):
        log_memory_usage(f"🧠 Epoch {epoch}/{EPOCHS}")
        losses = trainer.train_epoch(loader, epoch)
        print(f"✅ Epoch {epoch} losses: {losses}")

        # Save models after each epoch
        for name, model in models.items():
            save_model(model, name, epoch)

    # Final save of models
    for name, model in models.items():
        save_model(model, name)

    log_memory_usage("✅ Training Completed")
    print("✅ All final models saved.")

def load_all_models(checkpoint_dir: str = "checkpoints") -> dict:
    """
    Loads all trained deep learning models from disk.
    Includes encoder, VAE, SimCLR, and 3 scoring heads (risk_head, fraud_head, limit_head).
    """
    models = {}

    # Load encoder model
    encoder = DeepCreditEncoder()
    encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "encoder.pt"), map_location="cpu"))
    encoder.eval()
    models["encoder"] = encoder

    # Load VAE model
    vae = FraudVAE(input_dim=15, seq_len=12)
    vae.load_state_dict(torch.load(os.path.join(checkpoint_dir, "vae.pt"), map_location="cpu"))
    vae.eval()
    models["vae"] = vae

    # Load SimCLR model
    simclr = SimCLREncoder(input_dim=15, seq_len=12)
    simclr.load_state_dict(torch.load(os.path.join(checkpoint_dir, "simclr.pt"), map_location="cpu"))
    simclr.eval()
    models["simclr"] = simclr

    # Load risk classifier head
    risk_head = RiskClassifierHead(input_dim=128)
    risk_head.load_state_dict(torch.load(os.path.join(checkpoint_dir, "risk_head.pt"), map_location="cpu"))
    risk_head.eval()
    models["risk_head"] = risk_head

    # Load fraud detection head
    fraud_head = FraudDetectorHead(input_dim=64)
    fraud_head.load_state_dict(torch.load(os.path.join(checkpoint_dir, "fraud_head.pt"), map_location="cpu"))
    fraud_head.eval()
    models["fraud_head"] = fraud_head

    # Load loan limit regression head
    limit_head = LoanLimitRegressor(input_dim=128)
    limit_head.load_state_dict(torch.load(os.path.join(checkpoint_dir, "limit_head.pt"), map_location="cpu"))
    limit_head.eval()
    models["limit_head"] = limit_head

    return models

def run_full_scoring_pipeline(applicant):
    """
    Full scoring pipeline for an applicant. Calculates final score, loan limit, fraud score, etc.
    """
    sequence_tensor = extract_sequence_tensor(applicant).unsqueeze(0)  # [1, 12, 15]
    salary_seq = sequence_tensor[:, :, :5]
    credit_seq = sequence_tensor[:, :, 5:]

    models = load_all_models()
    encoder = models["encoder"]
    vae = models["vae"]
    simclr = models["simclr"]
    risk_head = models["risk_head"]
    fraud_head = models["fraud_head"]
    limit_head = models["limit_head"]

    with torch.no_grad():
        # 🔐 Deep embedding
        embedding = encoder(salary_seq, credit_seq)

        # 🧠 Risk score from deep classifier head (0-100)
        risk_score = risk_head(embedding)
        final_score = float(torch.clamp(risk_score * 100.0, 0.0, 100.0).item())

        # 💰 Loan limit prediction (regression)
        loan_limit = int(torch.clamp(limit_head(embedding), 0).item())

        # 🔍 VAE anomaly detection
        recon, mu, logvar = vae(sequence_tensor)
        vae_anomaly = float(torch.mean((recon - sequence_tensor) ** 2).item())

        # 🛡️ SimCLR + fraud head
        simclr_embedding = simclr(sequence_tensor)
        fraud_score = float(fraud_head(simclr_embedding).item())

        # 🔐 Confidence & Alert
        confidence_level = (
            "HIGH" if final_score >= 70 and vae_anomaly < 0.2 else
            "LOW" if final_score < 40 or vae_anomaly > 0.6 else
            "MEDIUM"
        )
        fraud_alert = "HIGH" if fraud_score > 0.5 else "LOW"

        # 🧠 Segment ID (can later be KMeans clustered)
        segment_id = "SEG_A1"

        # ✅ Approval decision
        approval_decision = None
        if final_score >= 70 and fraud_score < 0.3 and vae_anomaly < 0.2:
            approval_decision = "APPROVED"
        elif final_score < 50 or vae_anomaly > 0.6 or fraud_score > 0.5:
            approval_decision = "REJECTED"

        # 🧾 Human-readable explanation
        summary_parts = [
            f"Risk score is {final_score:.2f},",
            f"anomaly level is {'low' if vae_anomaly < 0.2 else 'moderate' if vae_anomaly < 0.6 else 'high'},",
            f"fraud signal is {'low' if fraud_score < 0.3 else 'elevated'}.",
            f"Loan limit: {loan_limit:,.0f} UZS."
        ]
        if approval_decision == "APPROVED":
            summary_parts.insert(0, "Applicant is likely eligible.")
        elif approval_decision == "REJECTED":
            summary_parts.insert(0, "Applicant is likely not eligible.")
        else:
            summary_parts.insert(0, "Applicant is under further review.")
        human_summary = " ".join(summary_parts)

        # 📊 Salary pattern
        salary_records = applicant.get("salary_records", [])
        if salary_records:
            recent_salaries = [float(r.get("salary", 0)) for r in salary_records[-6:] if float(r.get("salary", 0)) > 0]
            avg_salary = np.mean(recent_salaries) if recent_salaries else 0
            salary_analysis = (
                f"Average salary over last 6 periods: {avg_salary:,.0f} UZS. "
                f"{'Stable' if np.std(recent_salaries) < avg_salary * 0.25 else 'Fluctuating'} income trend."
            )
        else:
            salary_analysis = "No salary history available."

        # 📊 Credit behavior
        credit_records = applicant.get("credit_records", [])
        credit_analysis = "No credit data found."
        for r in credit_records:
            try:
                raw = r.get("credit_data", r)
                if isinstance(raw, str):
                    raw = json.loads(raw)
                contracts = raw.get("report", {}).get("contingent_liabilities", {}).get("contingent_liability", [])
                if not isinstance(contracts, list):
                    contracts = [contracts]
                overdue_count = sum(1 for c in contracts if float(c.get("overdue_debt_sum", 0)) > 0)
                total_contracts = len(contracts)
                credit_analysis = f"{total_contracts} contracts found. {overdue_count} had overdue payments."
            except Exception:
                credit_analysis = "Credit analysis could not be completed."

        return ScoringResponse(
            pinfl=applicant.get("pinfl"),
            final_score=round(final_score, 2),
            loan_limit=loan_limit,
            fraud_score=round(fraud_score, 4),
            vae_anomaly=round(vae_anomaly, 4),
            confidence_level=confidence_level,
            segment_id=segment_id,
            fraud_alert=fraud_alert,
            approval_decision=approval_decision,
            explanations={
                "vae_mse": round(vae_anomaly, 4),
                "simclr_std": round(fraud_score, 4)
            },
            human_readable_summary=human_summary,
            salary_analysis=salary_analysis,
            credit_analysis=credit_analysis
        )

if __name__ == "__main__":
    train()
