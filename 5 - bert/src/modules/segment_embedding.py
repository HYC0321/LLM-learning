import torch
import torch.nn as nn
import math

class SegmentEmbedding(nn.Module):
    """
    分段嵌入层。
    
    用于区分输入中的两个不同句子。
    '词汇表'大小为2 (句子A, 句子B)。
    """
    def __init__(self, d_model: int, vocab_size: int = 2):
        """
        参数:
            d_model (int): 嵌入向量的维度，必须与词元嵌入和位置嵌入的维度相同。
            vocab_size (int): 分段的数量，对于BERT通常是2。
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        参数:
            segment_ids (torch.Tensor): 分段ID张量，形状为 [batch_size, seq_len]，
                                        其中的值通常为 0 或 1。
        返回:
            torch.Tensor: 分段嵌入向量，形状为 [batch_size, seq_len, d_model]。
        """
        return self.embedding(segment_ids)
