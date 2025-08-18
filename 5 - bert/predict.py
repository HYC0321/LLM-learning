from json import load
import torch
import config
import random
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 设置与加载
# 在config.py中配置
MODEL_DIR = config.MODEL_DIR

print(f"使用的设备: {config.DEVICE}")
print(f"正在从 '{config.MODEL_DIR}' 加载模型和分词器...")

# 加载我们微调后保存的分词器和模型
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(config.DEVICE)
    model.eval()
    print("模型和分词器加载成功！")
except EnvironmentError:
    print(f"错误：找不到模型或分词器文件。请确保 '{MODEL_DIR}' 目录存在且包含训练好的文件。")
    exit()

# 定义标签映射
LABEL_MAP = {0: "负面 (Negative)", 1: "正面 (Positive)"}

# 2. 编写 predict 函数
def predict(text: str):
    """
    接收一段文本，并预测其情感。
    
    参数:
        text (str): 需要分析的英文句子。
        
    返回:
        str: 预测的情感标签 ("正面" 或 "负面")。
    """
    with torch.no_grad():
        # 1. 使用分词器处理输入文本
        #    return_tensors="pt" 会将输出直接转换为 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # 2. 将处理好的数据移动到与模型相同的设备
        inputs = {k: v.to(config.DEVICE) for k,v in inputs.items()}

        # 3. 将数据输入模型进行前向传播
        outputs = model(**inputs)

        # 4. 获取模型的输出 logits
        logits = outputs.logits

        # 5. 通过 argmax 获取概率最高的类别索引
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
        # 6. 使用标签映射将索引转换为可读的标签
        return LABEL_MAP[predicted_class_id]


if __name__ == '__main__':
    # 定义一些测试句子
    print("\n--- 正在加载 SST-2 验证集 ---")
    raw_datasets = load_dataset("glue", "sst2")
    validation_dataset  = raw_datasets['validation']

    # 为了避免每次都看同样的样本，我们随机抽取
    num_samples_to_show = 10
    # 生成随机索引
    random_indices = random.sample(range(len(validation_dataset)), num_samples_to_show)

    print(f"\n--- 随机抽取 {num_samples_to_show} 个验证集样本进行测试 ---")

    correct_predictions = 0
    for i in random_indices:
        sample = validation_dataset[i]
        sentence = sample['sentence']
        true_label_id = sample['label']
        true_label_text = LABEL_MAP[true_label_id]

        # 使用我们的 predict 函数进行预测
        predicted_label_text = predict(sentence)

        # 检查预测是否正确
        is_correct = (predicted_label_text == true_label_text)
        if is_correct:
            correct_predictions += 1

        # 打印结果
        print(f"\n----------- 样本 {i} -----------")
        print(f"句子: '{sentence}'")
        print(f"真实情感: {true_label_text}")
        print(f"模型预测: {predicted_label_text}")
        print(f"结果: {'✔️ 正确' if is_correct else '❌ 错误'}")

    print("\n" + "="*50)
    print("测试完成！")
    print(f"在抽样的 {num_samples_to_show} 个样本中，模型正确预测了 {correct_predictions} 个。")
    print(f"样本准确率: {correct_predictions / num_samples_to_show * 100:.2f}%")
        
