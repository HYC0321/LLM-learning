import torch
import torch.nn as nn

from .token_embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .segment_embedding import SegmentEmbedding

class BERTInput(nn.modules):
    """
    BERT 输入模块，负责将三种嵌入合并。
    """
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
                
