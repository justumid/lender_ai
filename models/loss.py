import torch
import torch.nn.functional as F


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent contrastive loss for SimCLR.
    """
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(len(z), dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -9e15)

    positives = torch.cat([torch.diag(sim, len(z_i)), torch.diag(sim, -len(z_i))])
    loss = -positives + torch.logsumexp(sim, dim=1)
    return loss.mean()


def vae_loss(recon_x, x, mu, logvar, beta=1.0, reduction="mean"):
    """
    β-VAE loss: reconstruction + β * KL divergence
    """
    if reduction == "mean":
        recon = F.mse_loss(recon_x, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        recon = F.mse_loss(recon_x, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + beta * kl
