import torch
import torch.nn.functional as F
from src.modules.transformer import Transformer
from datasets import load_dataset
# from torchtext.datasets import Multi30k
import config
import spacy
from torchmetrics.text import BLEUScore
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="Some child DataPipes are not exhausted when __iter__ is called.*",)

def greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, device):
    """
    贪心解码函数
    """
    model.eval() # 切换到评估模式
    with torch.no_grad():
        # 通过编码器获得 memory
        memory = model.encoder(model.pos_encoder(model.src_embedding(src)), src_mask)

        # 初始化解码器的输入，只包含一个 <sos> token
        # 形状为 [1, 1] (Batch_size=1, Seq_len=1)
        tgt_ids = torch.full((1,1), sos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # 获取目标序列的 padding mask (在这里其实都是1，没有padding)
            tgt_padding_mask = (tgt_ids != model.pad_idx).int().unsqueeze(1)

            # Decoder 在内部会处理 look-ahead mask
            decoder_output = model.decoder(model.pos_encoder(model.tgt_embedding(tgt_ids)), memory, tgt_padding_mask, src_mask)

            # 获取最后一个时间步的 logits
            last_logits = model.generator(decoder_output[:, -1])

            # 找到概率最高的词的ID
            next_word_id = last_logits.argmax(1)

            # 将新生成的词拼接到目标序列中
            tgt_ids = torch.cat([tgt_ids, next_word_id.unsqueeze(0)], dim=1)

            # 如果生成了 <eos> token，则停止解码
            if next_word_id.item() == eos_idx:
                break

    return tgt_ids.squeeze(0)

def beam_search_decode(model, src, src_mask, max_len, sos_idx, eos_idx, device, beam_width=3):
    """
    束搜索解码函数 (简化版)
    """
    model.eval()

    with torch.no_grad():
        # 通过编码器获得 memory
        memory = model.encoder(model.pos_encoder(model.src_embedding(src)), src_mask)

        # 初始化：一个包含 <sos> 的序列，分数为 0
        # beams 列表存储元组 (sequence, log_probability_score)
        beams = [(torch.full((1, 1), sos_idx, dtype=torch.long, device=device), 0.0)]
        completed_beams = []

        for _ in range(max_len - 1):
            new_beams = []
            # 如果序列已结束，则移入完成列表
            for seq, score in beams:
                if seq[0, -1].item() == eos_idx:
                    completed_beams.append((seq, score))
                    continue
                
                tgt_padding_mask = (seq != model.pad_idx).int().unsqueeze(1)
                decoder_output = model.decoder(model.pos_encoder(model.tgt_embedding(seq)), memory, tgt_padding_mask, src_mask)

                last_logits = model.generator(decoder_output[:, -1])

                # 转换为 log 概率
                log_probs = F.log_softmax(last_logits, dim=-1)

                # 获取 top-k 的 log 概率和对应的索引
                top_log_probs, top_ids = torch.topk(log_probs, beam_width, dim=1)

                # 扩展当前的束
                for i in range(beam_width):
                    next_id = top_ids[0, i].reshape(1, 1)
                    log_prob = top_log_probs[0, i].item()

                    new_seq = torch.cat([seq, next_id], dim=1)
                    new_score = score + log_prob
                    new_beams.append((new_seq, new_score))

            # 如果所有束都已完成，则提前退出
            if not new_beams:
                break

            # 从所有候选者中选出 top-k
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # 将仍在进行的束也加入完成列表
        completed_beams.extend(beams)
        # 找到分数最高的完成束 (这里可以引入长度惩罚等高级策略)
        best_beam = sorted(completed_beams, key=lambda x: x[1], reverse=True)[0]

        return best_beam[0].squeeze(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_model_40.pt"
SRC_VOCAB_PATH = "models/src_vocab.pt"
TGT_VOCAB_PATH = "models/tgt_vocab.pt"


# --- 主翻译函数 ---
def translate_sentence(sentence: str, model, src_vocab, tgt_vocab, device, max_len=50, decode_strategy='greedy'):
    model.eval()

    # 1. 使用英语分词器处理英语输入
    spacy_en = spacy.load('en_core_web_sm')
    tokens = [tok.text for tok in spacy_en.tokenizer(sentence)]
    src_tokens = [src_vocab['<sos>']] + src_vocab(tokens) + [src_vocab['<eos>']]
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    src_mask = (src_tensor != src_vocab['<pad>']).int().unsqueeze(1)

    # 2. 选择解码策略
    if decode_strategy == 'greedy':
        result_ids = greedy_decode(model, src_tensor, src_mask, max_len, tgt_vocab['<sos>'], tgt_vocab['<eos>'], device)
    elif decode_strategy == 'beam':
        result_ids = beam_search_decode(model, src_tensor, src_mask, max_len,
                                        tgt_vocab['<sos>'], tgt_vocab['<eos>'], device, beam_width=3)
    else:
        raise ValueError("解码策略必须是 'greedy' 或 'beam'")

    # 3. 将 ID 转换回文本
    # 忽略 <sos> 和 <eos>
    result_tokens = tgt_vocab.lookup_tokens(result_ids.tolist())
    return "".join(result_tokens).replace("<sos>", "").replace("<eos>", "").replace("<unk>","*").strip()

def evaluate_and_show_examples(model, test_data_iterator, src_vocab, tgt_vocab, device, num_examples=5):
    """
    翻译测试集中的一些样本并打印对比结果。
    """
    print("\n--- 正在展示部分翻译样本 ---")

    # 从迭代器中取出样本
    examples = list(test_data_iterator)[:num_examples]
    for i, (src_raw, tgt_raw) in enumerate(examples):
        greedy_translation = translate_sentence(src_raw, model, src_vocab, tgt_vocab, DEVICE, decode_strategy='greedy')
        beam_translation = translate_sentence(src_raw, model, src_vocab, tgt_vocab, DEVICE, decode_strategy='beam')
    
        print(f"\n----------- 样本 {i+1} -----------")
        print(f"源句 (DE): {src_raw}")
        print(f"目标 (EN): {tgt_raw}")
        print(f"贪心 (EN): {greedy_translation}")
        print(f"束搜 (EN): {beam_translation}")

def calculate_bleu(model, test_data_iterator, src_vocab, tgt_vocab, device):
    """
    在整个测试集上计算 BLEU 分数。
    """
    print("\n--- 正在计算整个测试集的 BLEU 分数 ---")
    targets = []
    predictions = []
    bleu_metric = BLEUScore()

    for src_raw, tgt_raw in tqdm(test_data_iterator, desc="Calculating BLEU"):
        prediction = translate_sentence(src_raw, model, src_vocab, tgt_vocab, device)
        # torchmetrics 需要 list of predictions 和 list of list of targets
        predictions.append(prediction)
        targets.append([tgt_raw])

    # 计算 BLEU 分数
    bleu_score = bleu_metric(predictions, targets)
    return bleu_score.item()

if __name__ == '__main__':

    # --- 加载模型和词汇表 ---
    print("正在加载模型和词汇表...")

    # 1. 加载已保存的词汇表对象
    src_vocab = torch.load(SRC_VOCAB_PATH)
    tgt_vocab = torch.load(TGT_VOCAB_PATH)


    # 2. 从检查点中恢复超参数来实例化模型
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    print(f"最优模型保存于第{checkpoint['epoch']}轮训练")

    model = Transformer(
        src_vocab_size=checkpoint['src_vocab_size'],
        tgt_vocab_size=checkpoint['tgt_vocab_size'],
        d_model=config.D_MODEL,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF
    ).to(DEVICE)

    # 3. 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载 Multi30k 测试集(测试集有问题，用验证集代替)
    raw_datasets = load_dataset("Aye10032/zh-en-translate-20k")
    # 这是一个关键的“适配器”函数，
    # 它将 Hugging Face dataset 的格式转换为我们后续代码期望的简单元组迭代器格式。
    def hf_dataset_to_iterator(dataset, lang_pair=("english", "chinese")):
        src_lang, tgt_lang = lang_pair
        for item in dataset:
            yield (item[src_lang], item[tgt_lang])
    # test_data = Multi30k(split=('valid'))
    test_data = hf_dataset_to_iterator(raw_datasets['train'])

    # 打印一些翻译样本进行直观感受
    evaluate_and_show_examples(model, test_data, src_vocab, tgt_vocab, DEVICE, num_examples=20)

    # 在整个测试集上计算量化指标
    # 注意：计算 BLEU 会遍历整个测试集，可能需要一些时间
    test_data = hf_dataset_to_iterator(raw_datasets['validation'])
    test_bleu_score = calculate_bleu(model, test_data, src_vocab, tgt_vocab, DEVICE)
    
    print("-" * 50)
    print(f"✅ 评估完成！")
    print(f"整个测试集上的 BLEU 分数: {test_bleu_score * 100:.2f}")
    
# --- 执行翻译 ---
# test_data = Multi30k(split='test')


# test_sentence_de = "Ein Mann in einem blauen T-Shirt sitzt auf einer Bank."

# print("\n--- 翻译测试 ---")
# print(f"源句 (DE): {test_sentence_de}")

# greedy_translation = translate_sentence(test_sentence_de, model, src_vocab, tgt_vocab, DEVICE, decode_strategy='greedy')
# print(f"贪心解码 (EN): {greedy_translation}")

# beam_translation = translate_sentence(test_sentence_de, model, src_vocab, tgt_vocab, DEVICE, decode_strategy='beam')
# print(f"束搜索解码 (EN): {beam_translation}")




