import os
import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
# --- 1. 导入所有必要的模块和类 ---
# 获取模型、数据管道、学习率调整器、训练引擎、评估引擎
from src.modules.transformer import Transformer
# from data_loader import train_dataloader, valid_dataloader, src_vocab, tgt_vocab
from data_loader_zh import train_dataloader, valid_dataloader, src_vocab, tgt_vocab
from src.train.scheduler import CustomLRScheduler
from src.train.engine import train_one_epoch, evaluate

# --- 2. 设置配置和超参数 ---
import config

# --- 3. 主训练函数 ---
def main():
    writer = SummaryWriter("./log")

    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab)

    print("初始化模型...")
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=config.D_MODEL,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        pad_idx=config.PAD_IDX,
        dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQ_LEN
    ).to(config.DEVICE)
    
    # 定义优化器 (使用论文中建议的参数)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    scheduler = CustomLRScheduler(optimizer, config.D_MODEL, config.WARMUP_STEPS)

    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    # 添加模型保存逻辑
    best_valid_loss = float('inf') # 初始化一个无穷大的最佳损失值
    saved_model_path = "models"
    os.makedirs(saved_model_path, exist_ok=True) # 确保保存目录存在

    print("开始训练...")
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        # 训练一个周期
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, criterion, config.DEVICE)

        end_time = time.time()

        # 评估模型
        valid_loss = evaluate(model, valid_dataloader, criterion, config.DEVICE)

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        cur_lr = scheduler.get_last_lr()[0]

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
        print(f"\tTrain Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f}")
        print(f"\tCurrent LR: {cur_lr:.8f}")

        writer.add_scalar("train_loss", train_loss, epoch+1)
        writer.add_scalar("valid_loss", valid_loss, epoch+1)
        writer.add_scalar("LR", cur_lr, epoch+1)

        # 检查是否需要保存模型 ---
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            # 创建检查点字典
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_valid_loss,
                'src_vocab_size': SRC_VOCAB_SIZE,
                'tgt_vocab_size': TGT_VOCAB_SIZE
            }
            
            # 保存检查点
            torch.save(checkpoint, os.path.join(saved_model_path, f'best_model_{config.NUM_EPOCHS}.pt'))
            print(f"✔️ Validation loss improved to {best_valid_loss:.4f}. Saving best model...")

    # 使用 torch.save 保存 vocab 对象
    torch.save(src_vocab, os.path.join(saved_model_path, 'src_vocab.pt'))
    torch.save(tgt_vocab, os.path.join(saved_model_path, 'tgt_vocab.pt'))

    print(f"训练完成。最佳模型已保存在 '{saved_model_path}/best_model_{config.NUM_EPOCHS}.pt'")
    print(f"词汇表已保存在 '{saved_model_path}/src_vocab.pt' 和 '{saved_model_path}/tgt_vocab.pt'")


if __name__ == '__main__':
    main()






