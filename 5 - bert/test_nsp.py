import torch

from src.modules.bert_input import BERTInput
from src.modules.bert_encoder import BERTEncoder
from src.modules.nsp_head import NSPHead



# --- 调试与验证 ---
# 超参数设置
VOCAB_SIZE = 20000
D_MODEL = 768
NUM_LAYERS = 12
NUM_HEADS = 12
D_FF = 3072
MAX_LEN = 256
BATCH_SIZE = 32
SEQ_LEN = 128
PAD_IDX = 0
CLS_IDX = 101 # 假设 [CLS] 的 ID 是 101

# 1. 实例化所有模块
bert_input = BERTInput(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=MAX_LEN)
bert_encoder = BERTEncoder(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF
)
nsp_head = NSPHead(d_model=D_MODEL)

# 2. 创建假数据
# 词元ID，确保每句话的开头都是 [CLS]
fake_token_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
fake_token_ids[:, 0] = CLS_IDX 
# 分段ID
fake_segment_ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
fake_segment_ids[:, SEQ_LEN // 2:] = 1
# 填充掩码
attention_mask = (fake_token_ids != PAD_IDX).int().unsqueeze(1)

print("--- 输入形状 ---")
print(f"Token IDs 形状: {fake_token_ids.shape}")
print("-" * 40)


# --- 3. 运行完整流程 ---
# 步骤 A: 获取输入嵌入
input_embeddings = bert_input(fake_token_ids, fake_segment_ids)
# 步骤 B: 将嵌入送入 BERT Encoder
encoder_output = bert_encoder(input_embeddings, attention_mask)
# 步骤 C: 将 Encoder 输出送入 NSP Head
nsp_logits = nsp_head(encoder_output)

print("--- 输出形状 ---")
print(f"Encoder 输出形状: {encoder_output.shape}")
print(f"NSP Head 输出 (Logits) 形状: {nsp_logits.shape}")
print("-" * 40)


# --- 4. 验证 ---
expected_shape = (BATCH_SIZE, 2)
assert nsp_logits.shape == expected_shape, "NSP Logits 输出形状不匹配！"

print("✅ 验证成功！NSP 任务的整个前向传播流程已正确实现。")