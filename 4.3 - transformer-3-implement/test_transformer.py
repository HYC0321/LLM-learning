from src.modules.transformer import Transformer
import torch
import torch.nn as nn

# --- 调试设置 ---
# 假设我们的源语言和目标语言共享词汇表
# 都有 1000 个词
SRC_VOCAB_SIZE = 10000
TGT_VOCAB_SIZE = 12000
# 模型的维度 (embedding 维度)
D_MODEL = 512
# PFFN的中间维度
D_FF = 2048
# 拆分头树
NUM_HEADS = 8
# Encoder 层数
NUM_LAYERS = 6
# Dropout概率
DROPOUT = 0.1
# 批次大小
BATCH_SIZE = 32
# 源和目标序列长度
# 目标序列长度可以与源序列不同
SRC_SEQ_LEN = 100
TGT_SEQ_LEN = 80
# 最大序列长度
MAX_SEQ_LEN = 5000

# --- 实例化完整的 Transformer 模型 ---
print("正在实例化 Transformer 模型...")
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    pad_idx=0,
    dropout=DROPOUT,
    max_seq_len=MAX_SEQ_LEN
)

print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
# total = 0
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         num_params = param.numel()
#         print(f"{name:50s} : {num_params:,}")
#         total += num_params

# print(f"\n总可训练参数量: {total:,}")
print("-" * 50)

# --- 创建假数据 ---
print("正在创建模拟数据...")
src_ids = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN))
tgt_ids = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN))

# --- 运行模型进行一次前向传播 ---
print("正在执行一次前向传播...")
logits = model(src_ids, tgt_ids)


# --- 验证输出 ---
print("-" * 50)
print(f"模型最终输出 Logits 形状: {logits.shape}")

expected_logits_shape = (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE)
assert logits.shape == expected_logits_shape, "最终 Logits 输出形状错误"

print("✅ 任务完成！最终的 Transformer 模型已成功组装并通过验证。")

