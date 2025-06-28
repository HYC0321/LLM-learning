import torch

# 1. 逐元素运算 (Element-wise Operations)：
print("--- Element-wise Operations ---")
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
print(f"Tensor A:\n{A}")
print(f"Tensor B:\n{B}")

print(f"A + B:\n{A + B}")
print(f"A - B:\n{A - B}")
print(f"A * B (element-wise):\n{A * B}")
print(f"A / B (element-wise):\n{A / B}")
print(f"torch.sqrt(A):\n{torch.sqrt(A)}")
print(f"torch.abs(torch.tensor([-1., 2.])):\n{torch.abs(torch.tensor([-1., 2.]))}")

# 2. 矩阵运算 (Matrix Operations)：
print("\n--- Matrix Operations ---")
C = torch.randn(2, 3) # 2x3
D = torch.randn(3, 4) # 3x4
print(f"Tensor C:\n{C}")
print(f"Tensor D:\n{D}")

matrix_mul = C @ D # 或 torch.matmul(C, D)
print(f"C @ D (Matrix Multiplication):\n{matrix_mul}")
print(f"Shape of C @ D: {matrix_mul.shape}")

E = torch.randn(3, 3)
print(f"Tensor E:\n{E}")
E_transpose = E.T # 或 E.transpose(0, 1)
print(f"E Transpose:\n{E_transpose}")

# 3. 聚合运算 (Aggregation Operations)：
print("\n--- Aggregation Operations ---")
F = torch.randn(4, 5)
print(f"Tensor F:\n{F}")

print(f"Sum of all elements in F: {torch.sum(F)}")
print(f"Mean of each row in F: {torch.mean(F, dim=1)}") # dim=1 表示对每行（跨列）求平均
print(f"Max of each column in F: {torch.max(F, dim=0).values}") # .values 获取最大值本身
print(f"Min of all elements in F: {torch.min(F)}")

# 4. 广播 (Broadcasting)：
print("\n--- Broadcasting ---")
G = torch.tensor([[1], [2], [3]]) # Shape: 3x1
H = torch.tensor([[10, 20, 30]]) # Shape: 1x3
print(f"Tensor G (shape {G.shape}):\n{G}")
print(f"Tensor H (shape {H.shape}):\n{H}")

sum_g_h = G + H
print(f"G + H (Broadcasting):\n{sum_g_h}")
print(f"Shape of G + H: {sum_g_h.shape}")
# 解释：G (3x1) 和 H (1x3) 进行加法时，
# G 会被扩展到 (3x3)，通过复制列 [1,2,3]
# H 会被扩展到 (3x3)，通过复制行 [10,20,30]
# 然后进行逐元素相加，得到 3x3 的结果。