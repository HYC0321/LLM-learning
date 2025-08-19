import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward import PositionwiseFeedForward

class GPTBlock(nn.Module):
    """
    GPT 的核心构建模块。
    
    它由一个掩码多头自注意力层和一个前馈神经网络组成。
    本质上是 Transformer DecoderBlock 的简化版，移除了交叉注意力部分。
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

        # 第二个子层: 位置全连接前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            x (torch.Tensor): 输入张量，形状 [batch_size, tgt_len, d_model]。
            mask (torch.Tensor, optional): 输入张量的掩码 (结合了 look-ahead 和 padding)。
        
        返回:
            torch.Tensor: 输出张量，形状仍为 [batch_size, tgt_len, d_model]。
        """
        # --- 第一个子层: 带掩码的多头自注意力 ---
        # Q, K, V 都来自目标序列 tgt
        self_attn_output, _ = self.self_attn(x, x, x, mask)
        # 残差和归一化
        x = self.norm1(x + self.dropout1(self_attn_output))

        # --- 第二个子层: 位置全连接前馈网络 ---
        ff_output = self.feed_forward(x)
        # 残差和归一化
        output = self.norm2(x + self.dropout2(ff_output))

        return output

