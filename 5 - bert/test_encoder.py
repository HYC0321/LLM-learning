import torch

from src.modules.bert_input import BERTInput
from src.modules.bert_encoder import BERTEncoder


VOCAB_SIZE = 20000
D_MODEL = 768
NUM_LAYERS = 12 # BERT-base 的层数
NUM_HEADS = 12  # BERT-base 的头数
D_FF = 3072     # 4 * D_MODEL
MAX_LEN = 256
BATCH_SIZE = 32
SEQ_LEN = 128
PAD_IDX = 0     # 假设 <pad> 的 ID 是 0

# 1. 实例化输入模块和 BERT Encoder 模块
bert_input = BERTInput(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=MAX_LEN)
bert_encoder = BERTEncoder(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF
)

# 2. 创建假数据
fake_token_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
fake_segment_ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
# 创建填充掩码 (padding mask)
# BERT 的注意力机制不需要 look-ahead mask，只需要 padding mask
attention_mask = (fake_token_ids != PAD_IDX).int().unsqueeze(1)

print("--- 输入形状 ---")
print(f"Token IDs 形状: {fake_token_ids.shape}")
print(f"Segment IDs 形状: {fake_segment_ids.shape}")
print(f"Attention Mask 形状: {attention_mask.shape}")
print("-" * 40)


# --- 3. 运行完整流程 ---
# 步骤 A: 获取输入嵌入
input_embeddings = bert_input(fake_token_ids, fake_segment_ids)
print(f"BERTInput 输出 (嵌入) 形状: {input_embeddings.shape}")

# 步骤 B: 将嵌入送入 BERT Encoder
encoder_output = bert_encoder(input_embeddings, attention_mask)
print(f"BERTEncoder 输出 (最终表示) 形状: {encoder_output.shape}")
print("-" * 40)


# --- 4. 验证 ---
expected_shape = (BATCH_SIZE, SEQ_LEN, D_MODEL)
assert input_embeddings.shape == expected_shape, "BERTInput 输出形状不匹配！"
assert encoder_output.shape == expected_shape, "BERTEncoder 输出形状不匹配！"

print("✅ 验证成功！BERT 的核心 Encoder 栈已搭建完成，并且数据流形状正确。")