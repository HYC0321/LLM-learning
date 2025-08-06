from turtle import forward
import torch
import torch.nn as nn
from .token_embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    """
    一个完整的 Transformer 模型，封装了 Encoder, Decoder 和所有必要的层。
    这个版本适用于源和目标词汇表不同的情况。
    它实现了目标嵌入层和最终输出层的权重绑定。
    """
    def __init__(self, 
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 pad_idx: int = 0,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        super().__init__()

        # 定义源和目标嵌入层
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        # 定义位置编码层
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # 定义编码器和解码器
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout, max_seq_len)
        
        # 定义最后的输出线性层 (Generator)
        # 将 generator 的权重矩阵绑定目标侧嵌入层的权重矩阵
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)

        self.pad_idx = pad_idx # 获取 padding_idx 

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:
        """
        定义了从源和目标ID到最终输出Logits的完整流程。
        """
        # 1. 创建掩码
        # 源序列只需要 padding mask
        src_padding_mask = (src != self.pad_idx).int().unsqueeze(1)
        # 目标序列需要 padding mask，用于与 look-ahead mask 结合
        tgt_padding_mask = (tgt != self.pad_idx).int().unsqueeze(1)

        # 2. 对源序列进行编码
        src_embedded = self.pos_encoder(self.src_embedding(src))
        memory = self.encoder(src_embedded, src_padding_mask)

        # 3. 对目标序列进行解码
        # Decoder 的 forward 方法会在内部处理 look-ahead mask 的合并
        tgt_embedded = self.pos_encoder(self.tgt_embedding(tgt))
        decoder_output = self.decoder(tgt_embedded, memory, tgt_padding_mask, src_padding_mask)

        # 4. 生成最终输出 Logits
        logits = self.generator(decoder_output)

        return logits