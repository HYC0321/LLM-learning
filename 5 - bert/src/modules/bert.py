import torch
import torch.nn as nn

from .bert_input import BERTInput
from .bert_encoder import BERTEncoder
from .mlm_head import MLMHead
from .nsp_head import NSPHead

class BERTModel(nn.Module):
    """
    核心 BERT 模型，包含输入层和编码器栈。
    这是可以被预训练和微调任务复用的“基础模型”。
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.1, max_len=5000):
        super().__init__()
        self.bert_input = BERTInput(vocab_size, d_model, dropout, max_len)
        self.bert_encoder = BERTEncoder(num_layers, d_model, num_heads, d_ff, dropout)

    def forward(self, token_ids, segment_ids, attention_mask):
        # 获取输入嵌入
        x = self.bert_input(token_ids, segment_ids)
        # 通过编码器栈
        encoder_output = self.bert_encoder(x, attention_mask)
        return encoder_output
    
class BERT_Pretrain_Model(nn.Module):
    """
    用于预训练的完整 BERT 模型，包含核心 BERT 模型以及两个预训练任务的头部。
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        # 1. 实例化核心 BERT 模型
        self.bert = BERTModel(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)

        # 2. 实例化两个预训练任务的头部
        # MLM Head 需要 token embedding 权重进行绑定
        token_embedding_layer = self.bert.bert_input.token_embedding.embedding
        self.mlm_head = MLMHead(d_model, vocab_size, token_embedding_layer)
        self.nsp_head = NSPHead()

    def forward(self, token_ids, segment_ids, attention_mask, mlm_labels=None, nsp_labels=None):
        """
        前向传播并计算联合损失。
        """
        # 获取 Encoder 的输出
        encoder_output = self.bert(token_ids, segment_ids, attention_mask)

        # 通过两个头部得到 logits
        mlm_logits = self.mlm_head(encoder_output)
        nsp_logits = self.nsp_head(encoder_output)

        # 如果提供了标签，则计算损失
        if mlm_labels is not None and nsp_labels is not None:
            mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # MLM 标签中 -100 的位置被忽略
            nsp_loss_fn = nn.CrossEntropyLoss()

            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, self.bert.bert_input.token_embedding.embedding.num_embeddings), mlm_labels.view(-1))
            nsp_loss = nsp_loss_fn(nsp_logits.view(-1, 2), nsp_labels.view(-1))

            total_loss = mlm_loss + nsp_loss
            return total_loss, mlm_loss, nsp_loss, mlm_logits, nsp_logits
        else:
            return mlm_logits, nsp_logits

