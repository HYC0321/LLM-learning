from turtle import forward
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward import PositionwiseFeedForward


class EncoderBlock(nn.Module):
    """
    单个 Transformer Encoder 模块。
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
        # 第一个子层: 多头自注意力
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 第二个子层: 位置全连接前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            src (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]。
            src_mask (torch.Tensor, optional): 输入序列的掩码。
            
        返回:
            torch.Tensor: 输出张量，形状仍为 [batch_size, seq_len, d_model]。
        """
        # --- 第一个子层: 多头自注意力 + 残差和归一化 ---
        # 1. 计算自注意力输出
        # 注意：在自注意力中，Q, K, V 都来自同一个源 src
        attn_output, _ = self.self_attn(src, src, src, src_mask)

        # 2. 残差连接 (x + Sublayer(x))
        #    论文中 dropout 是作用在 sublayer 的输出上
        src = src + self.dropout1(attn_output)
        # 3. 层归一化
        x = self.norm1(src)

        # --- 第二个子层: 前馈网络 + 残差和归一化 ---
        # 1. 计算前馈网络输出
        ff_output = self.feed_forward(x)

        # 2. 残差连接
        x = x + self.dropout2(ff_output)

        # 3. 层归一化
        output = self.norm2(src)

        return output