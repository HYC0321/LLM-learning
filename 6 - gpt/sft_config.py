import torch

# --- 训练配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"  # 使用的模型，例如 "gpt2", "gpt2-medium"
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50

# --- 模型和数据配置 ---
# PAD_TOKEN_ID 将在 data_loader 中从 tokenizer 动态获取
PAD_TOKEN_ID = None 

# --- 推理配置 ---
MAX_LENGTH_GEN = 100 # 生成文本的最大长度

# --- 保存配置 ---
OUTPUT_DIR = "sft_model"