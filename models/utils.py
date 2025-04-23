import torch
import numpy as np
import random
import os
import psutil


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.
    """
    return tensor.detach().cpu().numpy()


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    Applies to torch, numpy, and random libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batch_to_device(batch, device: torch.device):
    """
    Move a batch of tensors to the specified device.
    Handles tuple or list batches.
    """
    if isinstance(batch, (list, tuple)):
        return [b.to(device) for b in batch]
    return batch.to(device)


def print_model_summary(model: torch.nn.Module, name: str = "Model"):
    """
    Print a summary of the model including parameter count.
    """
    print(f"\n🧠 {name} Summary:")
    print(model)
    print(f"Total trainable parameters: {count_parameters(model):,}\n")


def log_memory_usage(tag: str = "Memory") -> None:
    """
    Log current CPU and GPU memory usage.
    """
    process = psutil.Process(os.getpid())
    mem_cpu = process.memory_info().rss / (1024 * 1024)

    msg = f"🧠 {tag}: CPU = {mem_cpu:.2f} MB"

    if torch.cuda.is_available():
        mem_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
        max_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
        msg += f", GPU = {mem_gpu:.2f} MB (Max = {max_gpu:.2f} MB)"

    print(msg)
