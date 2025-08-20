import torch
from src.modules.gpt import GPT
from src.modules.utils import generate

# 超参数设置
VOCAB_SIZE = 10000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
BATCH_SIZE = 1 # 推理时 batch_size 通常为 1
SEQ_LEN = 10

# 1. 实例化未经训练的 GPT 模型
untrained_model = GPT(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF
)

# 2. 创建一个简单的 prompt
# 形状必须是 [batch_size, seq_len]，所以是 [[...]]
prompt = torch.tensor([[10, 25, 3, 1]]) # [1, 4]
print(f"原始 Prompt: {prompt}")
print(f"原始 Prompt 长度: {prompt.shape[1]}")

# 3. 设置生成参数并调用 generate 函数
MAX_NEW_TOKENS = 20
generated_sequence = generate(
    model=untrained_model,
    prompt_ids=prompt,
    max_new_tokens=MAX_NEW_TOKENS
)

print("\n--- 生成结果 ---")
print(f"生成的完整序列: {generated_sequence}")
print(f"生成的序列长度: {generated_sequence.shape[1]}")
print("-" * 40)

# 4. 验证
expected_len = prompt.shape[1] + MAX_NEW_TOKENS
assert generated_sequence.shape[1] == expected_len, "生成序列的长度不正确！"

print("✅ 验证成功！generate 函数的自回归循环机制工作正常。")