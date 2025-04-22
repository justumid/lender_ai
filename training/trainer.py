import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm
import torch.nn.functional as F


class DeepTrainer:
    def __init__(self,
                 models: Dict[str, torch.nn.Module],
                 optim: torch.optim.Optimizer,
                 loss_funcs: Dict[str, Any],
                 device: torch.device = torch.device("cpu")):
        """
        models: dict with keys ['encoder', 'risk', 'limit', 'fraud', 'vae', 'simclr']
        loss_funcs: dict with keys ['bce', 'mse']
        """
        self.models = models
        self.optim = optim
        self.loss_funcs = loss_funcs
        self.device = device

        for model in models.values():
            model.to(device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        for model in self.models.values():
            model.train()

        total_losses = {"risk": 0, "limit": 0, "vae": 0, "simclr": 0}
        for batch in tqdm(loader, desc="Training"):
            x, y_risk, y_limit = [b.to(self.device) for b in batch]
            salary = x[:, :, :5]
            credit = x[:, :, 5:]
            full_seq = x

            encoder = self.models['encoder']
            z = encoder(salary, credit)

            pred_risk = self.models['risk'](z)
            pred_limit = self.models['limit'](z)
            pred_fraud = self.models['fraud'](z)

            loss_risk = self.loss_funcs['bce'](pred_risk, y_risk)
            loss_limit = self.loss_funcs['mse'](pred_limit, y_limit)

            recon, mu, logvar = self.models['vae'](full_seq)
            vae_loss = F.mse_loss(recon, full_seq)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss_vae = vae_loss + kl

            z_simclr = self.models['simclr'](full_seq)
            sim_matrix = torch.matmul(z_simclr, z_simclr.T)
            mask = torch.eye(z_simclr.size(0), device=self.device).bool()
            simclr_loss = (sim_matrix[~mask].mean() - sim_matrix[mask].mean()) * 0.1

            total_loss = loss_risk + loss_limit + loss_vae + simclr_loss

            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            total_losses["risk"] += loss_risk.item()
            total_losses["limit"] += loss_limit.item()
            total_losses["vae"] += loss_vae.item()
            total_losses["simclr"] += simclr_loss.item()

        return {k: round(v, 4) for k, v in total_losses.items()}

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        for model in self.models.values():
            model.eval()

        total_risk_acc = 0
        total_samples = 0

        with torch.no_grad():
            for x, y_risk, _ in loader:
                x, y_risk = x.to(self.device), y_risk.to(self.device)
                salary, credit = x[:, :, :5], x[:, :, 5:]
                z = self.models['encoder'](salary, credit)
                preds = self.models['risk'](z)
                pred_labels = (preds > 0.5).float()
                total_risk_acc += (pred_labels == y_risk).sum().item()
                total_samples += y_risk.size(0)

        return {"risk_accuracy": round(total_risk_acc / total_samples, 4)}

