import torch
import torch.nn as nn
from .encoder import Encoder

class BERTEncoder(nn.Module):
    """
    BERT 的编码器主模块。
    
    这是一个简单的封装，主要是为了代码结构上的清晰。
    它接收 BERTInput 的输出，并将其传递给一个标准的 Transformer Encoder。
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):       
        """
        初始化 BERTEncoder。
        
        参数与标准的 Encoder 完全相同。
        """
        super().__init__()
        # 内部实例化一个我们已经实现的 Encoder
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x (torch.Tensor): 来自 BERTInput 模块的输入嵌入，
                                形状为 [batch_size, seq_len, d_model]。
            mask (torch.Tensor): 输入序列的填充掩码 (padding mask)，
                                 形状通常为 [batch_size, 1, seq_len]。
        
        返回:
            torch.Tensor: BERT Encoder 的输出，形状保持不变 [batch_size, seq_len, d_model]。
        """
        return self.encoder(x, mask)