import netrc
import torch
import torch.nn as nn

def generate(model: nn.Module,
             prompt_ids: torch.Tensor,
             max_new_tokens: int,
             device: str = 'cpu'):
    """
    使用贪心解码策略，自回归地生成文本。
    
    参数:
        model (nn.Module): 已经实例化的 GPT 模型。
        prompt_ids (torch.Tensor): 输入的 prompt 词元 ID 序列，
                                   形状为 [batch_size, seq_len]。
        max_new_tokens (int): 要生成的最大新词元数量。
        device (str): 计算设备 ('cpu' 或 'cuda')。
        
    返回:
        torch.Tensor: 包含了 prompt 和新生成内容的完整 ID 序列。
    """
    model.eval()
    model.to(device)
    current_ids = prompt_ids.to(device)

    # 使用 torch.no_grad() 关闭梯度计算，节省资源
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. 将当前序列传入模型
            # 模型的输出 logits 形状为 [batch_size, current_seq_len, vocab_size]
            logits = model(current_ids)

            # 2. 取最后一个位置的 logits
            # 这是对下一个词元的预测
            # 形状: [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]
            # 3. 使用 argmax 找到概率最高的词元
            # 形状: [batch_size]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            # 4. 将其 ID 拼接到序列末尾
            current_ids = torch.cat((current_ids, next_token_id.unsqueeze(1)), dim=-1)

            # (可选的停止条件) 如果需要，可以在这里检查 next_token_id 是否为 <eos> 符
            # if next_token_id.item() == EOS_IDX:
            #     break

    return current_ids




            
