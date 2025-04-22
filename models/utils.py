import torch
import numpy as np
import random


def count_parameters(model):
    """
    Count trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.
    """
    return tensor.detach().cpu().numpy()


def set_seed(seed: int = 42):
    """
    Set global random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(batch, device):
    """
    Move a batch of tensors to a given device.
    """
    return [b.to(device) for b in batch]
