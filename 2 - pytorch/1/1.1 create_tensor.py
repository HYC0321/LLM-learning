import torch

# 3x4 全零浮点型张量
zero_tensor = torch.zeros((3, 4), dtype=torch.float32)
print("Zero Tensor:")
print(zero_tensor)
print(f"Shape: {zero_tensor.shape}, Dtype: {zero_tensor.dtype}, Device: {zero_tensor.device}")

# 2x2x3 全一张量，整型
one_tensor = torch.ones(2, 2, 3, dtype=torch.int32)
print("\nOne Tensor:")
print(one_tensor)
print(f"Shape: {one_tensor.shape}, Dtype: {one_tensor.dtype}, Device: {one_tensor.device}")

# 5x5 随机浮点型张量 (均匀分布)
rand_float_tensor = torch.rand(5, 5)
print("\nRandom Float Tensor:")
print(rand_float_tensor)
print(f"Shape: {rand_float_tensor.shape}, Dtype: {rand_float_tensor.dtype}")

# 4x3 随机整型张量 (1到10之间)
rand_int_tensor = torch.randint(1, 11, (4, 3)) # 注意：第二个参数是上限（不包含）
print("\nRandom Integer Tensor:")
print(rand_int_tensor)
print(f"Shape: {rand_int_tensor.shape}, Dtype: {rand_int_tensor.dtype}")

data_tensor = torch.arange(1, 17)
print(data_tensor)
print(f"Shape: {data_tensor.shape}, Dtype: {data_tensor.dtype}")
data_tensor = data_tensor.reshape(2,2,2,2)
print(data_tensor)
print(f"Shape: {data_tensor.shape}, Dtype: {data_tensor.dtype}")