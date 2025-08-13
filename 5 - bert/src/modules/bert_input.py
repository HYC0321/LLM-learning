import torch
import torch.nn as nn

from .token_embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .segment_embedding import SegmentEmbedding

class BERTInput(nn.Module):
    """
    BERT 输入模块，负责将三种嵌入合并。
    """
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.segment_embedding = SegmentEmbedding(d_model)
        # 注意：我们的 PositionalEncoding 类本身就包含了加和操作和 dropout
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # 标准 BERT 实现中，在最后会有一个 LayerNorm 和 Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor):
        """
        参数:
            token_ids (torch.Tensor): 词元ID张量，形状 [batch_size, seq_len]。
            segment_ids (torch.Tensor): 分段ID张量，形状 [batch_size, seq_len]。
            
        返回:
            torch.Tensor: 合并后的输入表示，形状 [batch_size, seq_len, d_model]。
        """
        # 1. 获取词元嵌入和分段嵌入
        token_emb = self.token_embedding(token_ids)
        segment_emb = self.segment_embedding(segment_ids)

        # 2. 将词元嵌入和分段嵌入相加
        x = token_emb + segment_emb

        # 3. 将位置嵌入信息加入
        #    我们的 PositionalEncoding 类是在输入 x 的基础上直接加，所以可以直接调用
        x_with_pos = self.positional_encoding(x)

        # 4. 应用 LayerNorm 和 Dropout
        output = self.dropout(self.norm(x_with_pos))

        return output

