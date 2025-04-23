import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm
from models.utils import log_memory_usage  # ✅ import memory logger

class DeepTrainer:
    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        optim: torch.optim.Optimizer,
        loss_funcs: Dict[str, Any],
        device: torch.device = torch.device("cpu")
    ):
        """
        models: dict with keys ['encoder', 'vae', 'simclr']
        loss_funcs: dict with keys ['mse']
        """
        self.models = models
        self.optim = optim
        self.loss_funcs = loss_funcs
        self.device = device

        for model in models.values():
            model.to(device)

    def train_epoch(self, loader: DataLoader):
        for model in self.models.values():
            model.train()

        total_losses = {"vae": 0.0, "simclr": 0.0}
        log_memory_usage("🔄 Start of Epoch")  # ✅

        for batch_idx, (x,) in enumerate(tqdm(loader, desc="Training")):
            x = x.to(self.device)
            salary, credit = x[:, :, :5], x[:, :, 5:]

            encoder = self.models["encoder"]
            z = encoder(salary, credit)

            # VAE loss
            recon, mu, logvar = self.models["vae"](x)
            vae_loss = self._vae_loss(recon, x, mu, logvar)

            # SimCLR loss
            z_simclr = self.models["simclr"](x)
            simclr_loss = self._simclr_contrastive_loss(z_simclr)

            # Total loss
            total_loss = vae_loss + simclr_loss

            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            total_losses["vae"] += vae_loss.item()
            total_losses["simclr"] += simclr_loss.item()

            if batch_idx % 10 == 0:
                log_memory_usage(f"🧪 Batch {batch_idx}")  # ✅

        avg_losses = {k: round(v / len(loader), 4) for k, v in total_losses.items()}
        log_memory_usage("✅ End of Epoch")  # ✅
        return avg_losses

    def _vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        recon = F.mse_loss(recon_x, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl

    def _simclr_contrastive_loss(self, z, temperature=0.5):
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / temperature
        mask = torch.eye(len(z), dtype=torch.bool).to(z.device)
        sim = sim.masked_fill(mask, -9e15)
        positives = torch.cat([torch.diag(sim, len(z)//2), torch.diag(sim, -len(z)//2)])
        loss = -positives + torch.logsumexp(sim, dim=1)
        return loss.mean()
