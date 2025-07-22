from modules.token_embedding import TokenEmbedding
from modules.positional_encoding import PositionalEncoding
from modules.multi_head_attention import MultiHeadAttention
import torch
import torch.nn as nn

# --- 调试设置 ---
# 假设我们的词汇表里有 1000 个词
VOCAB_SIZE = 1000
# 模型的维度 (embedding 维度)
D_MODEL = 512
# 批次大小
BATCH_SIZE = 32
# 序列长度
SEQ_LEN = 60
# 拆分头树
NUM_HEADS = 8
# --- 实例化模块 ---
token_embedder = TokenEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
position_encoder = PositionalEncoding(d_model=D_MODEL)
multi_head_attention = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
# --- 创建假数据 ---
# 创建一个随机的 token ID 序列
# 它的值在 [0, VOCAB_SIZE-1] 之间
# 形状为 [BATCH_SIZE, SEQ_LEN]
fake_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

print(f"输入张量的形状: {fake_input_ids.shape}")
print("-" * 30)

# --- 运行流程 ---
# 1. Token Embedding
token_embeddings = token_embedder(fake_input_ids)
print(f"经过 TokenEmbedding 后的形状: {token_embeddings.shape}")
print(f"预期形状: (BATCH_SIZE, SEQ_LEN, D_MODEL) = ({BATCH_SIZE}, {SEQ_LEN}, {D_MODEL})")

# 2. Positional Encoding
position_encodings = position_encoder(token_embeddings)
print(f"经过 PositionalEncoding 后的形状: {position_encodings.shape}")
print(f"预期形状: (BATCH_SIZE, SEQ_LEN, D_MODEL) = ({BATCH_SIZE}, {SEQ_LEN}, {D_MODEL})")
print("-" * 30)

# 3. Muti-Head Attention
# 在自注意力 (self-attention) 场景中, Q, K, V 来自同一个源
# 形状: [BATCH_SIZE, SEQ_LEN, D_MODEL]
x = position_encodings
q = x
k = x
v = x
# 创建一个假的 padding mask
# 假设每个序列的最后 10 个 token 是 padding，需要被 mask 掉
# mask 中值为 1 的是有效部分，值为 0 的是 padding 部分
fake_mask = torch.ones(BATCH_SIZE, 1, SEQ_LEN)
fake_mask[:, :, -10:] = 0 # 将最后10个位置设为0

print(f"Q/K/V 输入形状: {q.shape}")
print(f"Mask 输入形状: {fake_mask.shape}")

output, attn_weights = multi_head_attention(q, k, v, fake_mask)

print(f"经过 MultiHeadAttention 后的形状: {output.shape}")
print(f"预期形状: (BATCH_SIZE, SEQ_LEN, D_MODEL) = ({BATCH_SIZE}, {SEQ_LEN}, {D_MODEL})")
print(f"注意力权重形状: {attn_weights.shape}")
print(f"预期形状: (BATCH_SIZE, SEQ_LEN, D_MODEL) = ({BATCH_SIZE}, {NUM_HEADS}, {SEQ_LEN}, {SEQ_LEN})")
print("-" * 30)

# --- 验证 ---
expected_shape = (BATCH_SIZE, SEQ_LEN, D_MODEL)
assert output.shape == expected_shape, \
    f"形状不匹配! 期望得到 {expected_shape}, 但实际得到 {output.shape}"

print("✅ 验证成功！输出张量的形状正确。")