from modules.token_embedding import TokenEmbedding
from modules.positional_encoding import PositionalEncoding

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

# --- 实例化模块 ---
token_embedder = TokenEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
position_encoder = PositionalEncoding(d_model=D_MODEL)
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
final_output = position_encoder(token_embeddings)
final_output = position_encoder(token_embeddings)
print(f"经过 PositionalEncoding 后的最终形状: {final_output.shape}")
print(f"预期形状: (BATCH_SIZE, SEQ_LEN, D_MODEL) = ({BATCH_SIZE}, {SEQ_LEN}, {D_MODEL})")
print("-" * 30)

# --- 验证 ---
expected_shape = (BATCH_SIZE, SEQ_LEN, D_MODEL)
assert final_output.shape == expected_shape, \
    f"形状不匹配! 期望得到 {expected_shape}, 但实际得到 {final_output.shape}"

print("✅ 验证成功！输出张量的形状正确。")