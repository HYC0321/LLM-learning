import torch
import torch.nn as nn

class MLMHead(nn.Module):
    """
    掩码语言模型头部。
    """
    def __init__(self, d_model: int, vocab_size: int, token_embedding_layer: nn.Embedding):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

        # 解码器层，负责将 d_model 维度的向量映射回词汇表大小
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        # 添加解码器的偏置项
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

        # --- 权重绑定 ---
        # 让解码器的权重与词元嵌入层的权重共享
        self.decoder.weight = token_embedding_layer.weight
        self.decoder.bias = self.decoder_bias

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        参数:
            encoder_output (torch.Tensor): 来自 BERTEncoder 的输出，形状 [B, S, D]。
        
        返回:
            torch.Tensor: MLM 的预测 logits，形状 [B, S, Vocab_Size]。
        """
        x = self.dense(encoder_output)
        x = self.activation(x)
        x = self.norm(x)
        logits = self.decoder(x)
        return logits

