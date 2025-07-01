import torch
import torch.nn as nn

# 假设有3个类别
num_classes = 3

# 1. 实例化 CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# 2. 模拟模型输出 (logits)
# batch_size=2, num_classes=3
# 例如：对于第一个样本，模型认为它属于类别0的可能性最高，类别2的可能性最低。
outputs = torch.tensor([[0.8, 0.1, 0.1],
                        [0.2, 0.7, 0.1]], requires_grad=True)

# 3. 模拟真实标签 (类别索引)
# 第一个样本的真实标签是类别0，第二个样本的真实标签是类别1
targets = torch.tensor([0, 1])

# 4. 计算损失
loss = criterion(outputs, targets)
print(f"交叉熵损失: {loss.item():.4f}")

# 5. 理解 Softmax 内部处理 (仅作演示，实际使用时无需手动计算)
softmax_outputs = torch.softmax(outputs, dim=1)
print(f"Softmax 输出概率:\n{softmax_outputs}")

# 手动计算验证 (与 CrossEntropyLoss 内部逻辑类似)
# loss_manual_1 = -torch.log(softmax_outputs[0, targets[0]])
# loss_manual_2 = -torch.log(softmax_outputs[1, targets[1]])
# mean_loss_manual = (loss_manual_1 + loss_manual_2) / 2
# print(f"手动计算损失: {mean_loss_manual.item():.4f}")  