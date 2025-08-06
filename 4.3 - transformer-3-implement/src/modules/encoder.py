import torch
import torch.nn as nn
from .encoder_block import EncoderBlock

class Encoder(nn.Module):
    """
    完整的 Transformer Encoder，由 N 个 EncoderBlock 堆叠而成。
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        参数:
            num_layers (int): EncoderBlock 的堆叠数量。
            d_model, num_heads, d_ff, dropout: 传递给 EncoderBlock 的参数。
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # 许多实现会在最后再加一个 LayerNorm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        参数:
            src (torch.Tensor): 经过 embedding 和位置编码后的输入，形状为 [batch_size, seq_len, d_model]。
            src_mask (torch.Tensor, optional): 输入序列的掩码。
            
        返回:
            torch.Tensor: Encoder 的最终输出，形状仍为 [batch_size, seq_len, d_model]。
        """
        output = src
        # 依次通过每个 EncoderBlock
        for layer in self.layers:
            output = layer(output, src_mask)

        # 应用最终的层归一化
        output = self.norm(output)
        return output