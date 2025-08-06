from turtle import forward
import torch
import torch.nn as nn
from .decoder_block import DecoderBlock


class Decoder(nn.Module):
    """
    完整的 Transformer Decoder，由 N 个 DecoderBlock 堆叠而成。
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, max_seq_len: int = 5000):
        """
        参数:
            d_model (int): 模型维度。
            num_heads (int): 注意力头数。
            d_ff (int): 前馈网络中间层维度。
            dropout (float): Dropout 概率。
            max_seq_len(int): 最大输入序列长度
        """
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        look_ahead_mask = torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=0)
        self.register_buffer('look_ahead_mask', look_ahead_mask)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_padding_mask: torch.Tensor = None,
                src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            tgt (torch.Tensor): 目标序列输入，形状 [batch_size, tgt_len, d_model]。
            memory (torch.Tensor): 来自 Encoder 的输出，形状 [batch_size, src_len, d_model]。
            tgt_padding_mask (torch.Tensor, optional): 目标序列的掩码(仅 padding) 会在 forward 中和 look-look_ahead 结合。
            src_mask (torch.Tensor, optional): 源序列的掩码 (仅 padding)。
        
        返回:
            torch.Tensor: 输出张量，形状仍为 [batch_size, tgt_len, d_model]。
        """
        tgt_len = tgt.size(1)
        look_ahead_mask = self.look_ahead_mask[ :tgt_len, :tgt_len]
        if tgt_padding_mask is not None:
            tgt_mask = torch.minimum(tgt_padding_mask, look_ahead_mask)
        else:
            tgt_mask = look_ahead_mask.unsqueeze(0)

        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, src_mask)

        output = self.norm(output)
        return output

        
