import torch
import spacy
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import warnings
import config


warnings.filterwarnings("ignore", category=UserWarning, message="Some child DataPipes are not exhausted*")

# 加载 spacy 语言模型
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    print("请先下载 spacy 语言模型: \npython -m spacy download de_core_news_sm\npython -m spacy download en_core_web_sm")
    exit()

# 定义分词函数
def tokenize_de(text):
    """对德语文本进行分词"""
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """对英语文本进行分词"""
    return [tok.text for tok in spacy_en.tokenizer(text)]

print("分词器加载成功。")

# --- 加载数据 ---
# split 参数可以是 ('train', 'valid', 'test')
train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'))

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

print("正在为德语（源语言）构建词汇表...")
src_vocab = build_vocab_from_iterator(
    yield_tokens(train_data, tokenize_de, 0),
    min_freq=2, # 一个词至少出现2次才会被加入词汇表
    specials=special_symbols, 
    special_first=True
)
src_vocab.set_default_index(UNK_IDX) # 设置默认索引，遇到未登录词时返回 <unk> 的索引

print("正在为英语（目标语言）构建词汇表...")
tgt_vocab = build_vocab_from_iterator(
    yield_tokens(train_data, tokenize_en, 1),
    min_freq=2,
    specials=special_symbols,
    special_first=True
)
tgt_vocab.set_default_index(UNK_IDX)

print("-" * 50)
print(f"源语言（德语）词汇表大小: {len(src_vocab)}")
print(f"目标语言（英语）词汇表大小: {len(tgt_vocab)}")

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
    src_transform = text_transform(src_vocab, tokenize_de, SOS_IDX, EOS_IDX)
    tgt_transform = text_transform(tgt_vocab, tokenize_en, SOS_IDX, EOS_IDX)

    for src_sample, tgt_sample in batch:
        src_batch.append(src_transform(src_sample))
        tgt_batch.append(tgt_transform(tgt_sample))

    # 使用 pad_sequence 对批次内的序列进行填充
    # batch_first=True 表示批次维度在前，即 [B, Seq_Len]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)

    return src_batch, tgt_batch

# --- 封装 DataLoader ---
train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

print(f"训练集批次长度: {len(train_dataloader)}")
print(f"验证集批次长度: {len(valid_dataloader)}")
print(f"测试集批次: {len(train_dataloader)}")

print("\nDataLoader 准备就绪。")