import torch
import spacy
import config
from datasets import load_dataset
# from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API.*")
import jieba

# 加载 spacy 语言模型
try:
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    print("请先下载 spacy 语言模型: \npython -m spacy download en_core_web_sm")
    exit()

# 定义分词函数
def tokenize_zh(text):
    """对中文文本进行分词"""
    return list(jieba.cut(text))

def tokenize_en(text):
    """对英语文本进行分词"""
    return [tok.text for tok in spacy_en.tokenizer(text)]

print("中英文分词器加载成功。")

# --- 加载数据 ---
try:
    raw_datasets = load_dataset("Aye10032/zh-en-translate-20k")
    # print(type(raw_datasets))
    # print(raw_datasets)
except Exception as e:
    print(f"加载数据集失败，可能是网络问题。错误信息: {e}")
    exit()

# 这是一个关键的“适配器”函数，
# 它将 Hugging Face dataset 的格式转换为我们后续代码期望的简单元组迭代器格式。
def hf_dataset_to_iterator(dataset, lang_pair=("english", "chinese")):
    src_lang, tgt_lang = lang_pair
    for item in dataset:
        yield (item[src_lang], item[tgt_lang])

# 创建可迭代的数据对象
train_data = hf_dataset_to_iterator(raw_datasets['train']) 
valid_data = hf_dataset_to_iterator(raw_datasets['validation'])
# test_data = hf_dataset_to_iterator(raw_datasets['test'])
# --- 定义特殊符号及其索引 ---
# <unk>：未知词汇
# <pad>：填充符号
# <sos>：句子起始符号
# <eos>：句子终止符号
UNK_IDX = config.UNK_IDX
PAD_IDX = config.PAD_IDX
SOS_IDX = config.SOS_IDX
EOS_IDX = config.EOS_IDX
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# --- 构建词汇表 ---
# tokens 生成器
def yield_tokens(data_iter, tokenizer, language_index):
    """一个辅助函数，用于从数据集中提取词元"""
    for data_sample in data_iter:
        yield tokenizer(data_sample[language_index])

# print("正在为德语（源语言）构建词汇表...")
# src_vocab = build_vocab_from_iterator(
#     yield_tokens(train_data, tokenize_de, 0),
#     min_freq=2, # 一个词至少出现2次才会被加入词汇表
#     specials=special_symbols, 
#     special_first=True
# )
# src_vocab.set_default_index(UNK_IDX) # 设置默认索引，遇到未登录词时返回 <unk> 的索引

print("正在为英语（源语言）构建词汇表...")
src_vocab = build_vocab_from_iterator(
    yield_tokens(train_data, tokenize_en, 0),
    min_freq=1,
    specials=special_symbols,
    special_first=True 
)
src_vocab.set_default_index(UNK_IDX)

# 注意：迭代器只能使用一次，因此在构建下一个词汇表前需要重新创建
train_data = hf_dataset_to_iterator(raw_datasets['train'])

print("正在为中文（目标语言）构建词汇表...")
tgt_vocab = build_vocab_from_iterator(
    yield_tokens(train_data, tokenize_zh, 1),
    min_freq=1,
    specials=special_symbols,
    special_first=True
)
tgt_vocab.set_default_index(UNK_IDX)


print("-" * 50)
print(f"源语言（英语）词汇表大小: {len(src_vocab)}")
print(f"目标语言（中文）词汇表大小: {len(tgt_vocab)}")

# --- 文本处理流水线 ---
def text_transform(vocab, tokenizer, sos_idx, eos_idx):
    def transform(text_sample):
        tokens = tokenizer(text_sample)
        # 将词元转换为 ID
        ids = vocab(tokens)
        # 添加 <SOS> 和 <EOS> 符号
        return torch.tensor([sos_idx] + ids + [eos_idx])
    return transform

# --- `collate_fn`：用于处理一个批次的数据 ---
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    src_transform = text_transform(src_vocab, tokenize_en, SOS_IDX, EOS_IDX)
    tgt_transform = text_transform(tgt_vocab, tokenize_zh, SOS_IDX, EOS_IDX)

    for src_sample, tgt_sample in batch:
        src_batch.append(src_transform(src_sample))
        tgt_batch.append(tgt_transform(tgt_sample))

    # 使用 pad_sequence 对批次内的序列进行填充
    # batch_first=True 表示批次维度在前，即 [B, Seq_Len]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)

    return src_batch, tgt_batch

# --- 封装 DataLoader ---

# 注意：迭代器只能使用一次，因此在构建 DataLoader 前需要重新创建
train_iter = hf_dataset_to_iterator(raw_datasets['train'])
valid_iter = hf_dataset_to_iterator(raw_datasets['validation'])

train_dataloader = DataLoader(list(train_iter), batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
valid_dataloader = DataLoader(list(valid_iter), batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_data, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

# print(f"训练集批次长度: {len(train_dataloader)}")
# print(f"验证集批次长度: {len(valid_dataloader)}")
# print(f"测试集批次: {len(train_dataloader)}")

print("\nDataLoader 准备就绪。")

# print("\n--- 正在检查一个批次的数据 ---")
# src_batch, tgt_batch = next(iter(train_dataloader))
# print(f"源语言（英语）批次形状: {src_batch.shape}")
# print(f"目标语言（中文）批次形状: {tgt_batch.shape}")

# # 检查一个样本
# sample_idx = 0
# src_sample_tokens = src_vocab.lookup_tokens(src_batch[sample_idx].tolist())
# tgt_sample_tokens = tgt_vocab.lookup_tokens(tgt_batch[sample_idx].tolist())
# print("-" * 50)
# print("样本 0 - 源语言 (英语):", src_sample_tokens)
# print("样本 0 - 目标语言 (中文):", tgt_sample_tokens)