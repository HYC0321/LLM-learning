import torch
import config
import os
from src.modules.gpt import GPT
from src.modules.utils import generate

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

model = GPT(
    vocab_size=vocab_size,
    d_model=config.D_MODEL,
    num_layers=config.NUM_LAYERS,
    num_heads=config.NUM_HEADS,
    d_ff=config.D_FF,
    dropout=config.DROPOUT,
    max_len=config.BLOCK_SIZE
).to(config.DEVICE)

output_dir = config.MODEL_DIR
check_point = torch.load(os.path.join(output_dir, f'shakespeare_model.pt'), map_location=config.DEVICE)
model.load_state_dict(check_point)

# 定义一个 prompt
prompt_text = "O Romeo, Romeo,"
# 将 prompt 编码为 ID
prompt_ids = torch.tensor([encode(prompt_text)], dtype=torch.long, device=config.DEVICE)

# 调用我们之前编写的 generate 函数
generated_ids = generate(
    model=model,
    prompt_ids=prompt_ids,
    max_new_tokens=100, # 生成 200 个新字符
    device=config.DEVICE
)

# 解码生成的 ID 序列
generated_text = decode(generated_ids[0].tolist())
print("--- 生成结果 ---")
print(generated_text)