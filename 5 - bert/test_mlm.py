import torch

from src.modules.bert_input import BERTInput
from src.modules.bert_encoder import BERTEncoder
from src.modules.mlm_head import MLMHead
from src.modules.utils import create_mlm_masks




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

# 定义特殊 token 的 ID
PAD_IDX = 0
CLS_IDX = 101
SEP_IDX = 102
MASK_IDX = 103

# 1. 实例化所有模块
bert_input = BERTInput(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=MAX_LEN)
bert_encoder = BERTEncoder(num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF)
# MLMHead 需要 token embedding 层的权重来进行绑定
mlm_head = MLMHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, 
                   token_embedding_layer=bert_input.token_embedding.embedding)

# 2. 创建假数据
fake_token_ids = torch.randint(4, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)) # 从 4 开始避免特殊token
fake_token_ids[:, 0] = CLS_IDX # 每句话开头是 CLS
fake_token_ids[:, SEQ_LEN // 2] = SEP_IDX # 中间放个 SEP
fake_segment_ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
fake_segment_ids[:, SEQ_LEN // 2 + 1:] = 1

# 3. 应用 MLM 遮盖
masked_ids, mlm_labels = create_mlm_masks(
    fake_token_ids, VOCAB_SIZE,
    pad_idx=PAD_IDX, mask_idx=MASK_IDX, 
    cls_idx=CLS_IDX, sep_idx=SEP_IDX
)
attention_mask = (masked_ids != PAD_IDX).int().unsqueeze(1)

print("--- 输入与标签形状 ---")
print(f"遮盖后的输入 ID 形状: {masked_ids.shape}")
print(f"MLM 标签形状: {mlm_labels.shape}")
print("-" * 40)

# --- 4. 运行完整流程 ---
input_embeddings = bert_input(masked_ids, fake_segment_ids)
encoder_output = bert_encoder(input_embeddings, attention_mask)
mlm_logits = mlm_head(encoder_output)

print("--- 输出形状 ---")
print(f"Encoder 输出形状: {encoder_output.shape}")
print(f"MLM Head 输出 (Logits) 形状: {mlm_logits.shape}")
print("-" * 40)

# --- 5. 验证 ---
expected_shape = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
assert mlm_logits.shape == expected_shape, "MLM Logits 输出形状不匹配！"

print("✅ 验证成功！MLM 任务的整个前向传播流程已正确实现。")