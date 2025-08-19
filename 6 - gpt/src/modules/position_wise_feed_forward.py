import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    位置全连接前馈网络。
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化 PositionwiseFeedForward。
        
        参数:
            d_model (int): 输入和输出的维度。
            d_ff (int): 中间层的维度，论文中建议为 d_model * 4。
            dropout (float): Dropout 的概率。
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        


    def forward(self, x: torch.Tensor):
        """
        前向传播。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]。
            
        返回:
            torch.Tensor: 输出张量，形状仍为 [batch_size, seq_len, d_model]。
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

