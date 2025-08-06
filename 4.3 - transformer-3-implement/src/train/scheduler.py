import torch

class CustomLRScheduler:
    """
    自定义学习率调度器，实现 Transformer 论文中的策略。
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """在每个训练步后调用，以更新学习率"""
        self.step_num += 1

        arg1 = self.step_num ** -0.5
        arg2 = self.step_num * (self.warmup_steps ** -1.5)

        # 计算新的学习率
        new_lr = (self.d_model ** -0.5) * min(arg1, arg2)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

