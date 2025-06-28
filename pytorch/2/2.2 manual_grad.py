import torch

print("\n--- 练习 2：手动梯度清零与累积 ---")

# 准备张量
data = torch.tensor(2.0, requires_grad=True)
weight = torch.tensor(3.0, requires_grad=True)
bias = torch.tensor(0.5, requires_grad=True)

print(f"初始张量：data={data}, weight={weight}, bias={bias}")

# 第一次前向传播与反向传播
print("\n--- 第一次前向传播与反向传播 ---")
output1 = data * weight + bias
loss1 = output1.sum()
loss1.backward()

print(f"第一次 backward 后：")
print(f"data.grad: {data.grad}")
print(f"weight.grad: {weight.grad}")
print(f"bias.grad: {bias.grad}")

# 第二次前向传播与反向传播 (不清零)
print("\n--- 第二次前向传播与反向传播 (不清零) ---")
output2 = data * weight + bias
loss2 = output2.sum()
loss2.backward()

print(f"第二次 backward 后 (未清零)：")
print(f"data.grad: {data.grad}")
print(f"weight.grad: {weight.grad}")
print(f"bias.grad: {bias.grad}")
print("观察：梯度累积了！每个张量的梯度值变成了第一次的两倍。")

# 手动清零梯度
print("\n--- 手动清零梯度 ---")
data.grad.zero_()
weight.grad.zero_()
bias.grad.zero_()

print(f"清零后：")
print(f"data.grad: {data.grad}")
print(f"weight.grad: {weight.grad}")
print(f"bias.grad: {bias.grad}")
print("确认：梯度已清零。")

# 第三次前向传播与反向传播 (清零后)
print("\n--- 第三次前向传播与反向传播 (清零后) ---")
output3 = data * weight + bias
loss3 = output3.sum()
loss3.backward()

print(f"第三次 backward 后 (清零后)：")
print(f"data.grad: {data.grad}")
print(f"weight.grad: {weight.grad}")
print(f"bias.grad: {bias.grad}")
print("观察：梯度恢复到第一次的值，因为在这次反向传播前进行了清零。")