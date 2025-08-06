import torch


# 1. reshape 与 view：
print("--- Reshape & View ---")
# 1维张量
flat_tensor = torch.arange(24)
print(f"Original flat tensor shape: {flat_tensor.shape}")

# reshape 成 2x3x4
reshaped_tensor_1 = flat_tensor.reshape(2, 3, 4)
print(f"Reshaped to (2, 3, 4) shape: {reshaped_tensor_1.shape}")

# view 成 4x6 (view 要求内存连续，通常 reshape 也可以，但 view 更严格)
viewed_tensor = flat_tensor.view(4, 6)
print(f"Viewed to (4, 6) shape: {viewed_tensor.shape}")

# 尝试用 -1 作为 reshape 的参数
reshaped_with_neg1 = flat_tensor.reshape(2, -1) # -1 会自动计算为 12
print(f"Reshaped to (2, -1) shape: {reshaped_with_neg1.shape}")
print(f"Reshaped to (-1, 3) shape: {flat_tensor.reshape(-1, 3).shape}") # -1 自动计算为 8

# 2. transpose 与 permute：
print("\n--- Transpose & Permute ---")
original_3d_tensor = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print(f"Original 3D tensor shape: {original_3d_tensor.shape}")

# 使用 transpose 交换第 0 维和第 2 维
transposed_tensor = original_3d_tensor.transpose(0, 2)
print(f"Transposed (0, 2) tensor shape: {transposed_tensor.shape}") # (4, 3, 2)

# 使用 permute 将维度顺序变为 (1, 2, 0)
permuted_tensor = original_3d_tensor.permute(1, 2, 0)
print(f"Permuted (1, 2, 0) tensor shape: {permuted_tensor.shape}") # (3, 4, 2)

# 3. squeeze 与 unsqueeze：
print("\n--- Squeeze & Unsqueeze ---")
sparse_tensor = torch.randn(1, 3, 1, 5)
print(f"Original sparse tensor shape: {sparse_tensor.shape}")

# 使用 squeeze 移除所有大小为 1 的维度
squeezed_tensor = sparse_tensor.squeeze()
print(f"Squeezed tensor shape: {squeezed_tensor.shape}") # (3, 5)

# 在移除维度后的张量的第 0 维和第 3 维分别使用 unsqueeze
# 注意：squeezed_tensor 已经是 (3, 5)
# 在第 0 维添加：(1, 3, 5)
unsqueeze_0 = squeezed_tensor.unsqueeze(0)
print(f"Unsqueeze at dim 0 shape: {unsqueeze_0.shape}")

# 在第 3 维添加：(3, 5, 1) (原始维度只有 0, 1，所以在 dim 3 添加会在末尾)
unsqueeze_3 = squeezed_tensor.unsqueeze(2) # 注意，如果要在 (3,5) 后加，应该是 dim=2
print(f"Unsqueeze at dim 2 shape: {unsqueeze_3.shape}")