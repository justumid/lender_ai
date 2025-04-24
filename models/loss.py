import torch
import torch.nn.functional as F


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent loss (Normalized Temperature-scaled Cross Entropy) for contrastive learning.

    Args:
        z_i (Tensor): Projections from view 1 (B, D)
        z_j (Tensor): Projections from view 2 (B, D)
        temperature (float): Temperature scaling for logits

    Returns:
        Tensor: Scalar contrastive loss
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T)  # cosine sim [2B, 2B]
    logits = similarity / temperature

    # Labels: positive pairs are (i, i + B)
    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    logits = logits.masked_fill(mask, -9e15)

    return F.cross_entropy(logits, labels)


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Beta-VAE loss = MSE reconstruction + β × KL divergence

    Args:
        recon_x (Tensor): Reconstructed output [B, ...]
        x (Tensor): Original input [B, ...]
        mu (Tensor): Latent mean [B, D]
        logvar (Tensor): Latent log variance [B, D]
        beta (float): KL scaling factor
        reduction (str): 'mean' or 'sum' for MSE

    Returns:
        Tensor: Scalar VAE loss
    """
    recon = F.mse_loss(recon_x, x, reduction=reduction)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    if reduction == "mean":
        kl = kl.mean()

    return recon + beta * kl
