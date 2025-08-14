from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

# 加载 GLUE 基准测试中的 SST-2 数据集
raw_datasets = load_dataset("glue", "sst2")

print("数据集加载成功！")
print(raw_datasets)
print(raw_datasets['train'].num_rows)

# 让我们看一个训练集中的样本
print("\n--- 训练集样本示例 ---")
sample = raw_datasets['train'][0]
print(sample)
print(f"句子: '{sample['sentence']}'")
print(f"标签: {sample['label']} (0: 负面, 1: 正面)")

# 指定我们要使用的预训练模型名称
model_checkpoint = "bert-base-uncased"

# 从 Hugging Face Hub 加载对应的分词器
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

print(f"\n分词器 '{model_checkpoint}' 加载成功！")

# 让我们看看 Tokenizer 能做什么
print("\n--- Tokenizer 功能演示 ---")
test_sentence = "This movie was fantastic!"
print(f"原始句子: '{test_sentence}'")

# 1. 分词 (Tokenize)
tokens = tokenizer.tokenize(test_sentence)
print(f"分词结果: {tokens}")

# 2. 将词元转换为 ID
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"ID 转换结果: {ids}")

# 3. 一步到位完成所有处理 (最常用)
#    会自动添加 [CLS] 和 [SEP]，并转换为 ID
encoded_input = tokenizer(test_sentence)
print("\n一步到位编码结果:")
for key, value in encoded_input.items():
    print(f"  {key}: {value}")
    
# 解码回原始词元，可以看到特殊符号
decoded_tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'])
print(f"\n解码后的 Input IDs: {decoded_tokens}")

# 定义预处理函数
def preprocess_function(examples):
    """
    对输入的文本样本进行分词、数值化并添加特殊符号。
    'examples' 是一个字典，key 是特征名，如 'sentence'。
    """
    # tokenizer 会自动处理分词、ID转换、添加特殊符号 [CLS] 和 [SEP]。
    # truncation=True 表示如果句子超过模型最大长度，则进行截断。
    return tokenizer(examples['sentence'], truncation=True)

print("预处理函数定义完成。")

# 使用 .map() 方法将预处理函数应用到所有数据分割上
# batched=True 参数会让 .map() 一次性向 preprocess_function 传入一批样本，
# 这样可以充分利用 tokenizer 的并行处理能力，速度非常快。
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

print("\n数据预处理完成！")

# 让我们看看数据集的变化，现在多了三个新列
print("\n--- 预处理后的数据集结构 ---")
print(tokenized_datasets)

print("\n--- 预处理后的一个样本 ---")
print(tokenized_datasets['train'][0])

# 1. 实例化 Data Collator
#    它会使用 tokenizer 的 padding token 来进行填充
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 2. 对数据集进行一些清理和格式化，为 DataLoader 做准备
#    移除不再需要的原始文本列
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
#    将 'label' 列重命名为 'labels'，这是很多 Hugging Face 模型期望的名称
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
#    设置数据集的格式为 PyTorch 张量
tokenized_datasets.set_format("torch")

# 3. 创建 DataLoader
BATCH_SIZE = 16 # 微调时通常使用较小的 batch size
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator
)

print("\nDataLoader 创建完成！")

# 4. 调试：检查一个批次的形状
print("\n--- 检查一个批次的数据 ---")
for batch in train_dataloader:
    break # 只取第一个批次
    
print({k: v.shape for k, v in batch.items()})
