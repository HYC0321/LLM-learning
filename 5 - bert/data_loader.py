import torch

def create_mlm_masks(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_prob: float = 0.15,
    pad_idx: int = 0,
    mask_idx: int = 103,
    cls_idx: int = 101,
    sep_idx: int = 102
):
    pass