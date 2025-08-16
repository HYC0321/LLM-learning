import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT = "bert-base-uncased"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # 微调时使用非常小的学习率
NUM_EPOCHS = 3        # 微调通常只需要 2-4 个 epoch


