import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding 模块。
    为输入的 embedding 向量注入位置信息。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        参数:
            d_model (int): embedding 向量的维度。
            dropout (float): Dropout 的概率。
            max_len (int): 预先计算位置编码的最大序列长度。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        # pe 的形状将是 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position 表示词在序列中的位置 (0, 1, 2, ...)
        # 形状为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term 用于计算 sin 和 cos 函数中的分母部分
        # 形状为 [d_model / 2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 使用 sin 给偶数维度赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        # 使用 cos 给奇数维度赋值
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 增加一个 batch 维度，使其能够与输入批次数据广播相加。
        # 最终 pe 形状为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # register_buffer 将 pe 注册为模型的缓冲区（buffer）。
        # 这意味着它将成为模型状态的一部分（例如，会被 model.to(device) 移动），
        # 但它不是一个模型参数，不会被优化器更新。
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x (torch.Tensor): 输入的 token embedding，形状为 [batch_size, seq_len, d_model]。
            
        返回:
            torch.Tensor: 加入了位置信息后的 embedding，形状仍为 [batch_size, seq_len, d_model]。
        """
        # x.size(1) 是序列的长度 (seq_len)。
        # 我们取出预先计算好的 pe 中对应长度的部分，并加到 x 上。
        # self.pe 的形状是 [1, max_len, d_model]，
        # self.pe[:, :x.size(1), :] 的形状是 [1, seq_len, d_model]，可以和 x 广播相加。
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)
    








