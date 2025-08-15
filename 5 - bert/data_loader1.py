import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding

def create_hf_dataloaders(model_checkpoint: str, batch_size: int):
    """
    使用 Hugging Face datasets 和 transformers 创建数据加载器。

    参数:
        model_checkpoint (str): 预训练模型的名称，例如 "bert-base-uncased"。
        batch_size (int): 批处理大小。

    返回:
        tuple: 包含 train_dataloader, eval_dataloader, tokenizer 的元组。
    """
    # --- 1. 加载 Tokenizer ---
    print(f"正在加载预训练分词器: {model_checkpoint}")
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

    # --- 2. 加载数据集 ---
    print("正在加载 GLUE/SST-2 数据集...")
    raw_datasets = load_dataset("glue", "sst2")

    # --- 3. 编写并应用预处理函数 ---
    def preprocess_function(examples):
        """对输入的文本样本进行分词和数值化。"""
        return tokenizer(examples['sentence'], truncation=True)
    
    print("正在对数据集进行预处理...")
    
