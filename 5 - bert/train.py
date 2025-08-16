import torch
import config
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_scheduler
from torchmetrics.classification import BinaryAccuracy
from tqdm.auto import tqdm
from data_loader import create_hf_dataloaders
import os

def main():
    # 1. 设置配置和超参数
    # 在config.py中配置
    print(f"使用的设备: {config.DEVICE}")

    # 2. 准备数据
    train_dataloader, eval_dataloader, tokenizer = create_hf_dataloaders(
        config.MODEL_CHECKPOINT,
        config.BATCH_SIZE
    )

    # 3. 加载模型
    print("正在加载预训练模型...")
    model = BertForSequenceClassification.from_pretrained(config.MODEL_CHECKPOINT, num_labels=2)
    model.to(config.DEVICE)

    # 4. 设置优化器和学习率调度器
    # 使用 AdamW 优化器
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 计算总的训练步数，用于学习率调度器
    num_training_steps = config.NUM_EPOCHS * len(train_dataloader)

    # 创建一个带预热的线性学习率调度器
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # 5. 设置评估指标
    accuracy_metric = BinaryAccuracy().to(config.DEVICE)

    # 6. 训练与验证循环
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(config.NUM_EPOCHS):
        # --- 训练 ---
        model.train()
        for batch in train_dataloader:
            # 将批次数据移动到指定设备
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}

            # 前向传播
            outputs = model(**batch)

            # 获取损失
            loss = outputs.loss

            # 反向传播
            loss.backward()

            # 更新参数和学习率
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
        
        # --- 验证 ---
        model.eval()
        accuracy_metric.reset() # 每个 epoch 后重置指标

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
                outputs = model(**batch)

                # 获取预测
                predictions = torch.argmax(outputs.logits, dim=-1)

                # 更新准确率计算
                accuracy_metric.update(predictions, batch['labels'])

        eval_accuracy = accuracy_metric.compute()

        print("-" * 50)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | 验证集准确率: {eval_accuracy.item():.4f}")
        print("-" * 50)

    print("微调训练完成！")

    # 保存最终的模型
    output_dir = "saved_models_sst2"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型和分词器已保存到 '{output_dir}'")

if __name__ == '__main__':
    main()
