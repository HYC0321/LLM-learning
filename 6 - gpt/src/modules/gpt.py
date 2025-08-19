import torch
import torch.nn as nn
from .token_embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .gpt_block import GPTBlock


class GPT(nn.Module):
    """
    完整的 GPT 模型架构。
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 num_layers: int, 
                 num_heads: int, 
                 d_ff: int, 
                 pad_idx: int = 0,
                 dropout: float = 0.1, 
                 max_len: int = 5000):
        super().__init__()
        self.pad_idx = pad_idx
        # 输入层
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # 掩码矩阵
        look_ahead_mask = torch.tril(torch.ones(max_len, max_len), diagonal=0)
        self.register_buffer('look_ahead_mask', look_ahead_mask)

        # 堆叠 N 个 GPTBlock
        self.layers = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # 输出前的层归一化
        self.norm = nn.LayerNorm(d_model)

        # 定义最后的输出线性层 (Generator)
        # 将 generator 的权重矩阵绑定目标侧嵌入层的权重矩阵
        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        self.generator.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        参数:
            input_ids (torch.Tensor): 输入的词元 ID 序列，形状 [batch_size, seq_len]。
        
        返回:
            torch.Tensor: 预测的 logits，形状 [batch_size, seq_len, vocab_size]。
        """

        # 1. 创建掩码
        input_len = input_ids.size(1)
        padding_mask = (input_ids != self.pad_idx).int().unsqueeze(1)
        look_ahead_mask = self.look_ahead_mask[ :input_len, :input_len]
        mask = torch.minimum(padding_mask, look_ahead_mask)

        # 2. 通过输入层
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        # 3. 通过所有 GPTBlock
        for layer in self.layers:
            x = layer(x, mask)

        # 4. 通过输出前的层归一化
        x = self.norm(x)

        # 5. 生成最终输出 Logits
        logits = self.generator(x)

        return logits






















        