from cProfile import label
from turtle import forward
import torch
import torch.nn as nn

from .bert import BERTModel


class BERTForSequenceClassification(nn.Module):
    """
    用于序列分类任务的 BERT 微调模型。
    """
    def __init__(self, bert_base_model: BERTModel, num_classes: int, dropout: float = 0.1):
        """
        参数:
            bert_base_model (BERTModel): 一个已经预训练好的核心 BERT 模型。
            num_classes (int): 分类的类别数量 (例如，情感二分类为 2)。
            dropout (float): 在分类器前应用的 dropout 概率。
        """
        super().__init__()
        self.bert = bert_base_model
        self.dropout = nn.Dropout(dropout)

        # 定义新的分类头
        # 我们将使用 [CLS] token 的输出来进行分类
        self.classifier = nn.Linear(self.bert.bert_encoder.encoder.layers[0].self_attn.d_model, num_classes)

    def forward(self, token_ids, segment_ids, attention_mask, labels=None):
        # 1. 通过核心 BERT 模型
        encoder_output = self.bert(token_ids, segment_ids, attention_mask)

        # 2. 提取 [CLS] token 的输出 (与 NSPHead 类似)
        # 形状: [batch_size, d_model]
        cls_output = encoder_output[:, 0]

        # 3. 应用 dropout 并进行分类
        cls_output_dropout = self.dropout(cls_output)
        logits = self.classifier(cls_output_dropout)

        # 如果提供了标签，则计算损失
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
            return loss, logits
        else:
            return logits
        