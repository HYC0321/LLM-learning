import torch
from src.modules.bert_input import BERTInput

# --- 调试与验证 ---
# 超参数设置
VOCAB_SIZE = 20000
D_MODEL = 768 # BERT-base 的维度
MAX_LEN = 256
BATCH_SIZE = 32
SEQ_LEN = 128

# 实例化模块
bert_input = BERTInput(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=MAX_LEN)

# 创建假数据
# 词元ID
fake_token_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
# 分段ID (例如，前一半是句子A，后一半是句子B)
fake_segment_ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
fake_segment_ids[:, SEQ_LEN // 2:] = 1

print("--- 输入形状 ---")
print(f"Token IDs 形状: {fake_token_ids.shape}")
print(f"Segment IDs 形状: {fake_segment_ids.shape}")
print("-" * 30)

# 运行模块
output = bert_input(fake_token_ids, fake_segment_ids)

# 验证输出
print("--- 输出形状 ---")
print(f"最终输出形状: {output.shape}")
print("-" * 30)

expected_shape = (BATCH_SIZE, SEQ_LEN, D_MODEL)
assert output.shape == expected_shape, "输出形状不匹配！"

print("✅ 验证成功！BERTInput 模块的输出张量形状正确。")