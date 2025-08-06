import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()# 设置模型为训练模式
    total_loss = 0
    num_batches = 0

    # 使用 tqdm 显示进度条
    for src, tgt in tqdm(dataloader, desc="Training"):
        num_batches += 1
        src = src.to(device)
        tgt = tgt.to(device)

        # 准备 decoder 输入和目标标签
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # 前向传播
        logits = model(src, tgt_input)

        # 重塑 logits 和 target 以匹配 CrossEntropyLoss 的期望输入
        # logits: [B, Tgt_Len-1, Vocab_Size] -> [B*(Tgt_Len-1), Vocab_Size]
        # target: [B, Tgt_Len-1] -> [B*(Tgt_Len-1)]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
        optimizer.step()

        # 更新学习率
        scheduler.step()

        total_loss += loss.item()

    return total_loss / num_batches

def evaluate(model, dataloader, criterion, device):
    model.eval() # 设置模型为评估模式
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            num_batches += 1
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()

    return total_loss / num_batches
