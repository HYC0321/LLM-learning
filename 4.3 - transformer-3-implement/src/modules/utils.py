import torch
import torch.nn as nn

def generate_square_subsequent_mask(size: int, device: str = 'cpu') -> torch.Tensor:
    mask = torch.tril(torch.ones(size, size, device=device), diagonal=0)
    return mask