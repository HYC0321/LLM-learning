from turtle import forward
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力模块
    """
    def __init__(self, dropout: float = 0.1):
        """
        初始化 ScaledDotProductAttention。
        
        参数:
            dropout (float): 在注意力权重上施加的 Dropout 概率。
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算缩放点积注意力。

        参数:
            q (torch.Tensor): 查询 Queries，形状为 [batch_size, num_heads, seq_len_q, d_k]。
            k (torch.Tensor): 键 Keys，形状为 [batch_size, num_heads, seq_len_k, d_k]。
            v (torch.Tensor): 值 Values，形状为 [batch_size, num_heads, seq_len_v, d_v]。
                            通常 seq_len_k == seq_len_v。
            mask (torch.Tensor, optional): 掩码，形状需能广播到 [batch_size, num_heads, seq_len_q, seq_len_k]。
                                        例如，可以是 [batch_size, 1, 1, seq_len_k] (padding mask) 或
                                        [batch_size, 1, seq_len_q, seq_len_k] (look-ahead mask)。
                                        Defaults to None.

        返回:
            (torch.Tensor, torch.Tensor):
                - output: 注意力机制的输出，形状为 [batch_size, num_heads, seq_len_q, d_v]。
                - attn_weights: 原始的（未经 dropout 的）注意力权重，形状为 [batch_size, num_heads, seq_len_q, seq_len_k]。
        """
        # 1. 计算 Q 和 K 的点积，并进行缩放
        d_k = k.size(-1)
        # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 应用掩码 (Masking)
        # 这是子任务的核心之一：在 mask 值为 0 (或 False) 的位置填充一个极小的负数。
        # 这样在经过 softmax 后，这些位置的概率会趋近于 0。
        if mask is not None:
            # `masked_fill_` 是一个原地操作，它会将 scores 中值为 True 的位置用 value 填充。
            # 我们希望在 mask 等于 0 的地方进行填充，所以使用 `mask == 0`。
            scores = scores.masked_fill(mask == 0, -1e9) # -1e9 是一个近似负无穷的数

        # 3. 对最后一个维度 (seq_len_k) 应用 softmax，得到注意力权重
        attn_weight = nn.functional.softmax(scores, dim=-1)

        # 4. 将权重做dropout处理后与 V 相乘
        # (..., seq_len_q, seq_len_k) @ (..., seq_len_v, d_v) -> (..., seq_len_q, d_v)
        # 注意：seq_len_k 必须等于 seq_len_v
        output = torch.matmul(self.dropout(attn_weight), v)

        return output, attn_weight





# def scaled_dot_product_attention(
#     self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     计算缩放点积注意力。

#     参数:
#         q (torch.Tensor): 查询 Queries，形状为 [batch_size, num_heads, seq_len_q, d_k]。
#         k (torch.Tensor): 键 Keys，形状为 [batch_size, num_heads, seq_len_k, d_k]。
#         v (torch.Tensor): 值 Values，形状为 [batch_size, num_heads, seq_len_v, d_v]。
#                           通常 seq_len_k == seq_len_v。
#         mask (torch.Tensor, optional): 掩码，形状需能广播到 [batch_size, num_heads, seq_len_q, seq_len_k]。
#                                        例如，可以是 [batch_size, 1, 1, seq_len_k] (padding mask) 或
#                                        [batch_size, 1, seq_len_q, seq_len_k] (look-ahead mask)。
#                                        Defaults to None.

#     返回:
#         (torch.Tensor, torch.Tensor):
#             - output: 注意力机制的输出，形状为 [batch_size, num_heads, seq_len_q, d_v]。
#             - attn_weights: 原始的（未经 dropout 的）注意力权重，形状为 [batch_size, num_heads, seq_len_q, seq_len_k]。
#     """
#     # 1. 计算 Q 和 K 的点积，并进行缩放
#     d_k = k.size(-1)
#     # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
#     scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
#     # 2. 应用掩码 (Masking)
#     # 这是子任务的核心之一：在 mask 值为 0 (或 False) 的位置填充一个极小的负数。
#     # 这样在经过 softmax 后，这些位置的概率会趋近于 0。
#     if mask is not None:
#         # `masked_fill_` 是一个原地操作，它会将 scores 中值为 True 的位置用 value 填充。
#         # 我们希望在 mask 等于 0 的地方进行填充，所以使用 `mask == 0`。
#         scores = scores.masked_fill(mask == 0, -1e9) # -1e9 是一个近似负无穷的数

#     # 3. 对最后一个维度 (seq_len_k) 应用 softmax，得到注意力权重
#     attn_weight = nn.functional.softmax(scores, dim=-1)

#     # 4. 将权重做dropout处理后与 V 相乘
#     # (..., seq_len_q, seq_len_k) @ (..., seq_len_v, d_v) -> (..., seq_len_q, d_v)
#     # 注意：seq_len_k 必须等于 seq_len_v
#     output = torch.matmul(self.dropout(attn_weight), v)

#     return output, attn_weight


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        初始化 MultiHeadAttention。

        参数:
            d_model (int): 模型的总维度。
            num_heads (int): 注意力头的数量。d_model 必须能被 num_heads 整除。
            dropout (float): Dropout 的概率。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头的维度

        # 定义 Q, K, V 的线性投影层
        # 注意：这里我们将 d_model 映射到 d_model，之后再进行切分
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 实例化 ScaledDotProductAttention 类, 定义注意力层
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # 定义最后的输出线性层
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 shape [batch_size, seq_len, d_model] 的张量切分成多头。
        返回 shape [batch_size, num_heads, seq_len, d_k] 的张量。
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头的输出合并。
        输入 shape [batch_size, num_heads, seq_len, d_k]，
        返回 shape [batch_size, seq_len, d_model] 的张量。
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数:
            q (torch.Tensor): 查询 Query，形状为 [batch_size, seq_len_q, d_model]。
            k (torch.Tensor): 键 Key，形状为 [batch_size, seq_len_k, d_model]。
            v (torch.Tensor): 值 Value，形状为 [batch_size, seq_len_v, d_model]。
            mask (torch.Tensor, optional): 掩码。Defaults to None.

        返回:
            (torch.Tensor, torch.Tensor):
                - output: 最终输出，形状为 [batch_size, seq_len_q, d_model]。
                - attn_weights: 注意力权重，形状为 [batch_size, num_heads, seq_len_q, seq_len_k]。
        """
        # 1. 线性投影
        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # 2. 切分成多头
        q_heads = self.split_heads(q_proj)
        k_heads = self.split_heads(k_proj)
        v_heads = self.split_heads(v_proj)

        if mask is not None:
            # 例如 mask shape 是 [batch, 1, seq_len]，需要扩展成 [batch, 1, 1, seq_len]
            # 以便和 scores [batch, h, q_len, k_len] 广播
            mask = mask.unsqueeze(1)

        # 3. 进行缩放点积注意力计算
        attention_output, attn_weights = self.attention(
            q_heads, k_heads, v_heads, mask
        )
        
        # 4. 合并多头
        combined_output = self.combine_heads(attention_output)

        # 5. 最终的线性变换
        output = self.w_o(combined_output)

        return output, attn_weights
        



        
