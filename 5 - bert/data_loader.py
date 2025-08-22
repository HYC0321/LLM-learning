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
        # tokenizer 会自动处理分词、ID转换、添加特殊符号 [CLS] 和 [SEP]。
        return tokenizer(examples['sentence'], truncation=True)
    
    print("正在对数据集进行预处理...")
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # --- 4. 创建 Data Collator ---
    # DataCollatorWithPadding 会自动处理批次内的动态填充
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 5. 清理和格式化数据集 ---
    # 移除模型不需要的列
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    # 将 'label' 列重命名为 'labels'
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # 设置数据集格式为 PyTorch 张量
    tokenized_datasets.set_format("torch")
    
    # --- 6. 创建 DataLoader ---
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator
    )

    print("DataLoader 创建完成！")

    return train_dataloader, eval_dataloader, tokenizer
