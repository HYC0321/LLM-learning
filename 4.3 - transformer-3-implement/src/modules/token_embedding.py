import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        参数:
            vocab_size (int): 词汇表的大小。
            d_model (int): embedding 向量的维度。
        """
        super().__init__()
        # nn.Embedding 是一个查找表，存储固定大小词汇表的 embedding。
        # 它将输入的索引（IDs）映射到稠密的向量。
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        参数:
            tokens (torch.Tensor): 输入的 token ID 序列，形状为 [batch_size, seq_len]。
        返回:
            torch.Tensor: embedding 后的向量，形状为 [batch_size, seq_len, d_model]。
        """
        # 乘以 sqrt(d_model) 是 "Attention Is All You Need" 论文中的一个细节。
        # 它有助于在与位置编码相加时，平衡两者的量级。
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)
