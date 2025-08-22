import torch

# --- 模型参数 ---
D_MODEL = 512
NUM_LAYERS = 8
NUM_HEADS = 8
D_FF = 2048
DROPOUT = 0.2

# --- 超参数 ---
BATCH_SIZE = 64
BLOCK_SIZE = 128 # 上下文长度 (seq_len)
NUM_EPOCHS = 5 # 训练周期数，可以根据需要增加
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用的设备: {DEVICE}")

MODEL_DIR = 'models'