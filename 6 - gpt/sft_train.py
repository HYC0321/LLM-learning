import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW

import sft_config as config
from sft_data_loader import create_sft_dataloaders


def main():
    # --- 1. 准备模型和分词器 ---
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    config.PAD_TOKEN_ID = tokenizer.pad_token_id # 更新配置中的 pad_id

    model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME).to(config.DEVICE)
    model.resize_token_embeddings(len(tokenizer)) # 确保模型嵌入层大小与分词器词汇表大小一致

    # --- 2. 准备数据 ---
    train_dataloader, validation_dataloader = create_sft_dataloaders(tokenizer, config.BATCH_SIZE)

    # --- 3. 设置优化器 ---
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # --- 4. 训练与验证循环 ---
    print("开始指令微调 (SFT)...")
    
    for epoch in range(config.NUM_EPOCHS):
        # --- 训练 ---
        model.train()
        total_loss = 0
        for input_ids, labels in train_dataloader:
            input_ids, labels = input_ids.to(config.DEVICE), labels.to(config.DEVICE)
            # 正向传播
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=(input_ids != model.config.pad_token_id))
            # 反向传播
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)

        # --- 验证 ---
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_ids, labels in validation_dataloader:
                input_ids, labels = input_ids.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=(input_ids != model.config.pad_token_id))
                loss = outputs.loss
                total_loss += loss.item()
        avg_validation_loss = total_loss / len(validation_dataloader)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_validation_loss:.4f}")

    print("指令微调完成！")

    # --- 5. 保存模型 ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"模型和分词器已保存到 '{config.OUTPUT_DIR}'")

if __name__ == '__main__':
    main()

        





