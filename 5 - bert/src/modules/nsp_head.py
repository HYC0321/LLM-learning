import torch
import torch.nn as nn

class NSPHead(nn.Module):
    """
    下一句预测头部。
    """
    def __init__(self, d_model: int):
        """
        参数:
            d_model (int): BERT Encoder 的输出维度。
        """
        super().__init__()
        # Pooler 部分
        self.pooler_dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

        # 最终的分类器
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        参数:
            encoder_output (torch.Tensor): 来自 BERTEncoder 的输出，
                                            形状为 [batch_size, seq_len, d_model]。
        
        返回:
            torch.Tensor: NSP 的预测 logits，形状为 [batch_size, 2]。
        """
        # 1. 提取 [CLS] 词元的输出向量
        # [CLS] 始终是序列的第一个词元，因此我们取索引为 0 的位置
        cls_token_output = encoder_output[:, 0] # 形状: [batch_size, d_model]

        # 2. 将 [CLS] 向量通过 Pooler 层
        pooled_output = self.activation(self.pooler_dense(cls_token_output))

        # 3. 将池化后的输出送入分类器
        nsp_logits = self.classifier(pooled_output)

        return nsp_logits