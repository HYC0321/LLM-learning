import torch

print("--- Normalization Challenge ---")# 标准化处理
X = torch.randn(10, 10) # 10x10 随机张量
print(f"Original X mean: {torch.mean(X):.4f}, std: {torch.std(X):.4f}")

# 计算均值和标准差
mu = torch.mean(X)
sigma = torch.std(X)

# 标准化
X_normalized = (X - mu) / sigma

print(f"Normalized X mean: {torch.mean(X_normalized):.4f}, std: {torch.std(X_normalized):.4f}")



print("\n--- Conditional Assignment Challenge ---")# 条件赋值
cond_tensor = torch.randint(0, 101, (5, 5)) # 0到100之间的随机整型张量
print(f"Original tensor for conditional assignment:\n{cond_tensor}")

# 将所有小于 20 的元素替换为 0
cond_tensor[cond_tensor < 20] = 0
print(f"Tensor after setting elements < 20 to 0:\n{cond_tensor}")

# 将所有大于 80 的元素替换为 100
cond_tensor[cond_tensor > 80] = 100
print(f"Tensor after setting elements > 80 to 100:\n{cond_tensor}")