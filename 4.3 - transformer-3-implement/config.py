import torch
# --- 训练超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 40
# 学习率初始值设为0，因为我们会使用自定义调度器覆盖它
LEARNING_RATE = 0 

# --- 模型超参数 ---
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048   # 4 * D_MODEL
DROPOUT = 0.1
MAX_SEQ_LEN = 5000

# --- 学习率调度器参数 ---
WARMUP_STEPS = 4000

# --- 特殊符号 ---
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3