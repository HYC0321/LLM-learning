import torch

def create_mlm_masks(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_prob: float = 0.15,
    pad_idx: int = 0,
    mask_idx: int = 103,
    cls_idx: int = 101,
    sep_idx: int = 102
):
    """
    为 MLM 任务创建遮盖后的输入和对应的标签。
    
    返回:
        masked_input_ids (torch.Tensor): 经过 80-10-10 规则处理后的输入 ID。
        mlm_labels (torch.Tensor): 对应的标签，未被遮盖的位置为 -100。
    """
    # 1. 创建标签张量，初始值和输入张量相同
    mlm_labels = input_ids.clone()

    # 2. 计算遮盖的概率矩阵
    # 我们不希望遮盖特殊 token (pad, cls, sep)，所以将它们对应位置的 mask 概率设为 0.0
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    special_tokens_mask = (input_ids == pad_idx) | \
                        (input_ids == cls_idx) | \
                        (input_ids == sep_idx)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # 3. 根据概率决定哪些 token 需要被遮盖
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 4. 在标签张量中，只保留被遮盖位置的原始 token ID，其他设为 -100 (这是 CrossEntropyLoss 默认的 ignore_index)
    mlm_labels[~masked_indices] = -100

    # 5. 对输入 ID 应用 80-10-10 规则
    masked_input_ids = input_ids.clone()

    # 80% 的情况: 替换为 [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    masked_input_ids[indices_replaced] = mask_idx

    # 10% 的情况: 随机替换
    # (先找出需要随机替换的位置)
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # (生成随机词)
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long)
    # (进行替换)
    masked_input_ids[indices_random] = random_words[indices_random]

    # 10% 的情况: 保持不变 (我们无需做任何事，因为 masked_input_ids 已经包含了原始值)

    return masked_input_ids, mlm_labels
