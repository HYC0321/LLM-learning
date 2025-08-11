# Transformer 机器翻译实现

这是一个基于 PyTorch 的 Transformer 模型完整实现，支持多种语言对的机器翻译任务。该项目实现了论文 "Attention Is All You Need" 中描述的 Transformer 架构，并提供了两种翻译任务的完整实现。

## 项目功能

- **完整的 Transformer 架构实现**：包括编码器、解码器、多头注意力机制、位置编码等核心组件
- **多语言对翻译支持**：
  - 德语到英语翻译（使用 Multi30k 数据集）
  - 中文到英语翻译（使用中英文翻译数据集）
- **多种解码策略**：支持贪心解码和束搜索解码
- **模型评估**：使用 BLEU 分数评估翻译质量
- **训练监控**：集成 TensorBoard 进行训练过程可视化
- **模型保存与加载**：自动保存最佳模型检查点
- **灵活的数据处理**：支持不同数据源和格式

## 项目结构

```
4.3 - transformer-3-implement/
├── src/
│   ├── modules/                    # 模型核心组件
│   │   ├── transformer.py         # 完整 Transformer 模型
│   │   ├── encoder.py             # 编码器
│   │   ├── decoder.py             # 解码器
│   │   ├── encoder_block.py       # 编码器层
│   │   ├── decoder_block.py       # 解码器层
│   │   ├── multi_head_attention.py # 多头注意力机制
│   │   ├── positional_encoding.py # 位置编码
│   │   ├── position_wise_feed_forward.py # 前馈网络
│   │   ├── token_embedding.py     # 词嵌入
│   │   └── utils.py               # 工具函数
│   └── train/                     # 训练相关模块
│       ├── engine.py              # 训练和评估引擎
│       └── scheduler.py           # 学习率调度器
├── models/                        # 保存的模型文件
│   ├── best_model_40.pt          # 中英翻译模型检查点
│   ├── best_model_80.pt          # 德英翻译模型检查点
│   ├── src_vocab.pt              # 源语言词汇表
│   └── tgt_vocab.pt              # 目标语言词汇表
├── log/                          # TensorBoard 日志
├── config.py                     # 配置文件
├── data_loader.py                # 德英翻译数据加载器
├── data_loader_zh.py             # 中英翻译数据加载器
├── train.py                      # 训练脚本
├── translate.py                  # 德英翻译推理脚本
├── translate_zh.py               # 中英翻译推理脚本
├── test_transformer.py           # 模型测试脚本
└── README.md                     # 项目说明文档
```

## 安装依赖

### 1. 安装 Python 依赖

在项目根目录（`LLM-learning`）下运行：

```bash
pip install -r requirements.txt
```

### 2. 下载语言模型

#### 对于德英翻译：
```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

#### 对于中英翻译：
```bash
python -m spacy download en_core_web_sm
pip install jieba  # 中文分词工具（如果未包含在 requirements.txt 中）
```

## 模型配置

主要超参数在 `config.py` 中配置：

- **模型参数**：
  - `D_MODEL = 512`：模型维度
  - `NUM_LAYERS = 6`：编码器/解码器层数
  - `NUM_HEADS = 8`：多头注意力头数
  - `D_FF = 2048`：前馈网络隐藏层维度
  - `DROPOUT = 0.1`：Dropout 概率

- **训练参数**：
  - `BATCH_SIZE = 128`：批次大小
  - `NUM_EPOCHS = 80`：训练轮数
  - `WARMUP_STEPS = 4000`：学习率预热步数

## 运行训练

### 1. 德语到英语翻译训练

```bash
cd "4.3 - transformer-3-implement"
python train.py
```

训练过程将：
- 自动下载 Multi30k 数据集
- 构建德语和英语词汇表
- 训练 Transformer 模型
- 保存最佳模型到 `models/best_model_80.pt`
- 生成 TensorBoard 日志到 `log/` 目录

### 2. 中文到英语翻译训练

如需训练中英翻译模型，需要修改 `train.py` 中的数据加载器导入：

```python
# 将这行：
from data_loader import train_dataloader, valid_dataloader, test_dataloader, src_vocab, tgt_vocab

# 改为：
from data_loader_zh import train_dataloader, valid_dataloader, test_dataloader, src_vocab, tgt_vocab
```

然后运行训练：

```bash
python train.py
```

### 3. 监控训练过程

使用 TensorBoard 查看训练进度：

```bash
tensorboard --logdir=log
```

然后在浏览器中访问 `http://localhost:6006`

## 运行推理

### 1. 德语到英语翻译

```bash
python translate.py
```

程序将加载训练好的德英翻译模型，您可以输入德语句子进行翻译。

### 2. 中文到英语翻译

```bash
python translate_zh.py
```

程序将：
- 加载训练好的中英翻译模型（`models/best_model_40.pt`）
- 自动展示 20 个翻译样本
- 计算整个验证集的 BLEU 分数
- 支持贪心解码和束搜索解码策略

### 3. 批量评估

两个翻译脚本都会自动在测试集上评估模型性能，计算 BLEU 分数并展示翻译样本对比。

## 测试模型

运行模型架构测试：

```bash
python test_transformer.py
```

这将验证模型的前向传播是否正常工作。

## 主要特性

### 1. 完整的 Transformer 实现
- 标准的编码器-解码器架构
- 多头自注意力和交叉注意力机制
- 位置编码和残差连接
- Layer Normalization

### 2. 多语言支持
- **德语-英语翻译**：使用 Multi30k 数据集，支持 spaCy 分词
- **中文-英语翻译**：使用中英翻译数据集，支持 jieba 中文分词
- 灵活的数据加载器设计，易于扩展到其他语言对

### 3. 高效的数据处理
- 使用 torchtext 和 Hugging Face datasets 进行数据预处理
- 支持动态批处理和填充
- 词汇表构建和序列化
- 多种数据源适配器

### 4. 灵活的解码策略
- **贪心解码**：快速生成，适合实时应用
- **束搜索解码**：更高质量的翻译，支持可配置的束宽度

### 5. 训练优化
- 自定义学习率调度器（论文中的预热策略）
- 梯度裁剪防止梯度爆炸
- 早停机制保存最佳模型
- TensorBoard 集成监控

## 性能指标

### 德语-英语翻译
- 使用 Multi30k 数据集训练
- 训练 80 轮后保存最佳模型
- 支持不同解码策略的性能对比

### 中文-英语翻译
- 使用 zh-en-translate-20k 数据集
- 训练 40 轮后保存最佳模型
- 自动计算验证集 BLEU 分数

## 数据集信息

### 德语-英语翻译
- **数据集**：Multi30k
- **规模**：约 30,000 个句子对
- **领域**：图像描述文本
- **分词**：使用 spaCy 德语和英语模型

### 中文-英语翻译
- **数据集**：zh-en-translate-20k (Hugging Face)
- **规模**：约 20,000 个句子对
- **分词**：中文使用 jieba，英文使用 spaCy
- **数据源**：`Aye10032/zh-en-translate-20k`

## 使用示例

### 德语翻译示例
```python
# 输入德语句子
test_sentence = "Ein Mann in einem blauen T-Shirt sitzt auf einer Bank."
# 输出英语翻译
# "A man in a blue T-shirt sits on a bench."
```

### 中文翻译示例
```python
# 输入英语句子
test_sentence = "A beautiful sunset over the mountains."
# 输出中文翻译（根据模型训练方向）
```

## 注意事项

1. **GPU 要求**：建议使用 GPU 进行训练，CPU 训练会非常缓慢
2. **内存需求**：确保有足够的内存加载数据集和模型
3. **训练时间**：完整训练可能需要数小时到数天，取决于硬件配置
4. **模型文件**：训练完成的模型文件较大，注意存储空间
5. **网络连接**：首次运行需要下载数据集，确保网络连接稳定

## 扩展功能

- **多语言对支持**：可以轻松扩展到其他语言对（如法语-英语、日语-英语等）
- **模型架构实验**：可以调整层数、注意力头数等超参数
- **解码策略扩展**：可以实现更复杂的解码算法（如核采样、top-p 采样）
- **评估指标扩展**：可以集成 ROUGE、METEOR 等其他评估指标
- **数据增强**：可以添加回译、同义词替换等数据增强技术

## 故障排除

1. **spaCy 模型下载失败**：
   ```bash
   # 手动下载
   python -m spacy download de_core_news_sm --user
   python -m spacy download en_core_web_sm --user
   ```

2. **CUDA 内存不足**：
   - 减小 `config.py` 中的 `BATCH_SIZE`
   - 或在 `config.py` 中设置 `DEVICE = "cpu"`

3. **数据集下载失败**：
   - 检查网络连接
   - 使用代理或 VPN
   - 手动下载数据集文件

4. **jieba 分词问题**：
   ```bash
   pip install jieba
   ```

5. **Hugging Face 数据集访问问题**：
   ```bash
   pip install datasets
   # 或设置 HF_ENDPOINT 环境变量
   ```

## 参考文献

- Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
