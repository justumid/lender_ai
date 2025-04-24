from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import time

class DeepTrainer:
    def __init__(self, models, optim, loss_funcs, device):
        self.models = models
        self.optim = optim
        self.loss_funcs = loss_funcs
        self.device = device
        self.writer = SummaryWriter(log_dir=Path("runs") / time.strftime("%Y%m%d-%H%M%S"))

    def train_epoch(self, train_loader, epoch):
        for model in self.models.values():
            model.train()

        epoch_losses = {"recon": 0.0, "fraud": 0.0, "risk": 0.0, "limit": 0.0, "total": 0.0}
        total_batches = len(train_loader)

        for batch_idx, (batch_x,) in enumerate(train_loader):
            batch_x = batch_x.to(self.device)

            # Forward pass: encoder
            encoder = self.models["encoder"]
            embedding = encoder(batch_x[:, :, :5], batch_x[:, :, 5:])  # → [B, 128]

            # VAE reconstruction loss
            recon, mu, logvar = self.models["vae"](batch_x)
            recon_loss = self.loss_funcs["mse"](recon, batch_x)

            # SimCLR fraud head loss
            simclr_embedding = self.models["simclr"](batch_x)  # → [B, 64]
            fraud_logits = self.models["fraud_head"](simclr_embedding)
            fraud_loss = F.binary_cross_entropy(fraud_logits, torch.ones_like(fraud_logits))

            # Risk head loss (target = 1 for now)
            risk_logits = self.models["risk_head"](embedding)
            risk_loss = F.binary_cross_entropy(risk_logits, torch.ones_like(risk_logits))

            # Loan limit head loss (target = 30M UZS)
            limit_pred = self.models["limit_head"](embedding)
            limit_loss = F.mse_loss(limit_pred, torch.ones_like(limit_pred) * 30_000_000)

            # Total loss
            total_loss = recon_loss + fraud_loss + risk_loss + limit_loss

            # Backward and optimize
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            # Logging
            epoch_losses["recon"] += recon_loss.item()
            epoch_losses["fraud"] += fraud_loss.item()
            epoch_losses["risk"] += risk_loss.item()
            epoch_losses["limit"] += limit_loss.item()
            epoch_losses["total"] += total_loss.item()

        # Average losses for the epoch
        for key in epoch_losses:
            epoch_losses[key] /= total_batches
            self.writer.add_scalar(f"Loss/{key}", epoch_losses[key], epoch)

        return epoch_losses
