import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU

# 1. 定义模型类
class Hyc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = Sequential(
            # 第一个线性层：从 input_dim 到 hidden_dim
            Linear(input_dim, hidden_dim),
            # 非线性激活
            ReLU(),
            # 第二个线性层：从 hidden_dim 到 output_dim
            Linear(hidden_dim, output_dim),
            # 非线性激活
            ReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
if __name__ == "__main__":

    # 2. 实例化模型
    input_dim = 10  # 输入特征的维度
    hidden_dim = 20 # 隐藏层的维度
    output_dim = 1  # 输出特征的维度

    hyc = Hyc(input_dim, hidden_dim, output_dim)
    print("模型结构:")
    print(hyc)

    # 3. 创建一个虚拟输入数据
    # batch_size 为 4，每个样本有 input_dim 个特征
    dummy_input = torch.randn(2, 3, input_dim)
    print("\n虚拟输入数据形状:", dummy_input.shape)

    # 4. 进行前向传播
    output = hyc(dummy_input)
    print("模型输出形状:", output.shape)