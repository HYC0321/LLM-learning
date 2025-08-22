import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def create_sft_dataloaders(tokenizer, batch_size):
    """
    创建指令微调所需的数据加载器。
    """
    # 我们的迷你指令数据集
    sft_data = [
        {"instruction": "中国的首都是哪里？", "input": "", "output": "中国的首都是北京。"},
        {"instruction": "将下面的英文翻译成中文。", "input": "The quick brown fox jumps over the lazy dog.", "output": "敏捷的棕色狐狸跳过了懒惰的狗。"},
        {"instruction": "写一首关于月亮的五言绝句。", "input": "", "output": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
        {"instruction": "根据下面的描述，判断情感是积极还是消极。", "input": "这部电影的特效令人惊叹，情节也非常感人。", "output": "积极"},
    ]

    # Prompt 模板
    def create_prompt(example):
        if example['input']:
            return f"指令：{example['instruction']}\n输入：{example['input']}\n回答："
        else:
            return f"指令：{example['instruction']}\n回答："
    
    class SFTDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            prompt = create_prompt(item)
            output = item['output']
            full_text = prompt + output + self.tokenizer.eos_token

            input_ids = self.tokenizer.encode(full_text)
            prompt_ids = self.tokenizer.encode(prompt)

            labels = input_ids.copy()
            labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

            return torch.tensor(input_ids), torch.tensor(labels)
    
    def collate_fn(batch):
        input_ids, labels = zip(*batch)
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        label_ids_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        return input_ids_padded, label_ids_padded
    
    sft_dataset = SFTDataset(sft_data, tokenizer)
    # 假设所有数据都用于训练，因为数据集非常小
    train_dataloader = DataLoader(
        sft_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
        )

    validation_dataloader = DataLoader(
        sft_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return train_dataloader, validation_dataloader
