import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward import PositionwiseFeedForward

class DecoderBlock(nn.Module):
    """
    单个 Transformer Decoder 模块。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数:
            d_model (int): 模型维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络中间层维度。
            dropout (float): Dropout 概率。
        """
        super().__init__()
        # 第一个子层: 带Look-Ahead Mask的多头自注意力
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 第二个子层: 交叉注意力 (Encoder-Decoder Attention)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 第三个子层: 位置全连接前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            tgt (torch.Tensor): 目标序列输入，形状 [batch_size, tgt_len, d_model]。
            memory (torch.Tensor): 来自 Encoder 的输出，形状 [batch_size, src_len, d_model]。
            tgt_mask (torch.Tensor, optional): 目标序列的掩码 (结合了 look-ahead 和 padding)。
            src_mask (torch.Tensor, optional): 源序列的掩码 (仅 padding)。
        
        返回:
            torch.Tensor: 输出张量，形状仍为 [batch_size, tgt_len, d_model]。
        """
        # --- 第一个子层: 带掩码的多头自注意力 ---
        # Q, K, V 都来自目标序列 tgt
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        # 残差和归一化
        x = self.norm1(tgt + self.dropout1(self_attn_output))

        # --- 第二个子层: 交叉注意力 ---
        # Query 来自前一层的输出 x，Key 和 Value 来自 Encoder 的 memory
        cross_attn_output, _ = self.cross_attn(x, memory, memory, src_mask)
        # 残差和归一化
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # --- 第三个子层: 位置全连接前馈网络 ---
        ff_output = self.feed_forward(x)
        # 残差和归一化
        output = self.norm3(x + self.dropout3(ff_output))

        return output

