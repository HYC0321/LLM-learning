import torch
from src.modules.gpt import GPT

# --- 调试与验证 ---
# 超参数设置
VOCAB_SIZE = 10000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
BATCH_SIZE = 32
SEQ_LEN = 128

# 1. 实例化 GPT 模型
model = GPT(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF
)

# 2. 创建假数据
fake_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

print("--- 输入形状 ---")
print(f"Input IDs 形状: {fake_input_ids.shape}")
print("-" * 40)

# 3. 运行模型
logits = model(fake_input_ids)

print("--- 输出形状 ---")
print(f"最终输出 Logits 形状: {logits.shape}")
print("-" * 40)

# 4. 验证
expected_shape = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
assert logits.shape == expected_shape, "输出形状不匹配！"

print("✅ 验证成功！MiniGPT 模型已搭建完成，并且数据流形状正确。")