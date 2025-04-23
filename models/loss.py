import torch
import torch.nn.functional as F


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent loss for SimCLR (normalized temperature-scaled cross entropy).
    
    Args:
        z_i (Tensor): Projection of view 1 (B, D)
        z_j (Tensor): Projection of view 2 (B, D)
        temperature (float): Temperature parameter

    Returns:
        Tensor: NT-Xent contrastive loss
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T)  # (2B, 2B)
    similarity_matrix = similarity_matrix / temperature

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ], dim=0).to(z.device)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    logits = similarity_matrix
    loss = F.cross_entropy(logits, labels)
    return loss


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    β-VAE loss: reconstruction error + β × KL divergence

    Args:
        recon_x (Tensor): reconstructed input [B, ...]
        x (Tensor): original input [B, ...]
        mu (Tensor): mean vector from encoder [B, D]
        logvar (Tensor): log variance vector from encoder [B, D]
        beta (float): KL divergence weighting factor
        reduction (str): 'mean' or 'sum'

    Returns:
        Tensor: total VAE loss
    """
    recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    if reduction == "mean":
        kl_div = kl_div.mean()

    return recon_loss + beta * kl_div
