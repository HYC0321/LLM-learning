import torch
import torch.nn as nn
import config
import os
from torch.utils.data import Dataset, DataLoader
from src.modules.gpt import GPT
from src.modules.utils import generate
from tqdm.auto import tqdm


# 1. (数据) 编写数据管道
print("正在加载数据并构建词汇表...")
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text[:300000]

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"语料库共有 {len(text)} 个字符，词汇表大小为 {vocab_size}。")
print(f"词汇表: {''.join(chars)}")


# 创建字符到索引 (stoi) 和索引到字符 (itos) 的映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 定义编码和解码函数
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# --- 创建 PyTorch Dataset ---
class ShakespeareDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.data = torch.tensor(encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        # 从数据中随机抓取一个长度为 block_size + 1 的片段
        chunk = self.data[idx: idx + self.block_size + 1]
        # 输入序列是前 block_size 个字符
        x = chunk[:-1]
        # 目标序列是后 block_size 个字符 (输入序列右移一位)
        y = chunk[1:]
        return x, y
    
# --- 实例化 Dataset 和 DataLoader ---
train_dataset = ShakespeareDataset(text, config.BLOCK_SIZE)
train_dataloader = DataLoader(train_dataset, config.BATCH_SIZE, True)

# 3. (编码) 实例化模型、损失函数和优化器
model = GPT(
    vocab_size=vocab_size,
    d_model=config.D_MODEL,
    num_layers=config.NUM_LAYERS,
    num_heads=config.NUM_HEADS,
    d_ff=config.D_FF,
    dropout=config.DROPOUT,
    max_len=config.BLOCK_SIZE
).to(config.DEVICE)

# 打印模型参数量
print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# 进度条记录
num_training_steps = config.NUM_EPOCHS * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

output_dir = config.MODEL_DIR
os.makedirs(output_dir, exist_ok=True)


print("\n--- 开始训练 ---")
model.train()
for epoch in range(config.NUM_EPOCHS):
    total_loss = 0
    for i, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

        # 前向传播
        logits = model(inputs)

        # 计算损失
        # 我们需要将 logits 和 targets 变形
        # logits: [B, Tgt_Len-1, Vocab_Size] -> [B*(Tgt_Len-1), Vocab_Size]
        # target: [B, Tgt_Len-1] -> [B*(Tgt_Len-1)]
        logits_reshaped = logits.view(-1, vocab_size)
        targets_reshaped = targets.view(-1)
        loss = criterion(logits_reshaped, targets_reshaped)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        progress_bar.update(1)
        # 打印训练进度
        if (i + 1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        
    print(f"--- Epoch {epoch+1} 完成, 平均 Loss: {total_loss / len(train_dataloader):.4f} ---")

print("训练完成！")

torch.save(model.state_dict(), os.path.join(output_dir, f'shakespeare_model.pt'))


# 5. (验证) 使用训练好的模型生成文本
print("\n--- 开始生成文本 ---")

# 定义一个 prompt
prompt_text = "O Romeo, Romeo,"
# 将 prompt 编码为 ID
prompt_ids = torch.tensor([encode(prompt_text)], dtype=torch.long, device=config.DEVICE)

# 调用我们之前编写的 generate 函数
generated_ids = generate(
    model=model,
    prompt_ids=prompt_ids,
    max_new_tokens=200, # 生成 200 个新字符
    device=config.DEVICE
)

# 解码生成的 ID 序列
generated_text = decode(generated_ids[0].tolist())

print("--- 生成结果 ---")
print(generated_text)