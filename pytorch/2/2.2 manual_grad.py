import torch

print("\n--- 练习 3：手动参数更新 ---")

# 定义数据和真实参数
x_data = torch.randn(10, 1) * 10
true_w = torch.tensor(2.0)
true_b = torch.tensor(1.0)
y_true = true_w * x_data + true_b + torch.randn(10, 1) * 0.5 # 增加一些噪声

print(f"真实的权重 true_w: {true_w.item():.4f}, 真实的偏置 true_b: {true_b.item():.4f}")

# 定义可学习参数
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)

print(f"初始权重 w: {w.item():.4f}, 初始偏置 b: {b.item():.4f}")

# 设置学习率和迭代次数
learning_rate = 0.01
num_iterations = 100

print(f"\n--- 训练循环（使用 .data）---")
# 训练循环（手动）
for i in range(num_iterations):
    # 前向传播
    y_pred = w * x_data + b

    # 计算损失 (MSE)
    loss = torch.mean((y_pred - y_true)**2)
    if i == 0:
        print(loss)

    # 梯度清零
    if w.grad is not None: # 第一次迭代时 grad 为 None
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    # 反向传播
    loss.backward()

    # 参数更新 (使用 .data)
    w.data = w.data - learning_rate * w.grad.data
    b.data = b.data - learning_rate * b.grad.data

    # 打印
    if (i + 1) % 10 == 0:
        print(f"迭代 {i+1:3d} | Loss: {loss.item():.4f} | w: {w.item():.4f} | b: {b.item():.4f}")

print("\n--- 训练结束（使用 .data）---")
print(f"最终权重 w: {w.item():.4f}, 最终偏置 b: {b.item():.4f}")
print("观察：w 和 b 的值应该非常接近 true_w 和 true_b。")