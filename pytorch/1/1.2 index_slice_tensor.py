import torch


# 创建 4x4 张量，元素从 1 到 16
data_tensor = torch.arange(1, 17).reshape(4, 4)
print("Original Tensor:")
print(data_tensor)

# 1. 基本索引：
print("\n--- Basic Indexing ---")
# 获取第一行 (索引为 0)
row_0 = data_tensor[0]
print(f"First row: {row_0}")

# 获取第三列 (索引为 2)
col_2 = data_tensor[:, 2] # : 表示所有行
print(f"Third column: {col_2}")

# 获取 (1, 2) (第二行第三列) 的元素
element_1_2 = data_tensor[1, 2]
print(f"Element at (1, 2): {element_1_2}")

# 2. 切片 (Slicing)：
print("\n--- Slicing ---")
# 获取第 2 到 3 行 (索引为 1 到 2)
rows_1_to_2 = data_tensor[1:3, :]
print(f"Rows 1 to 2: \n{rows_1_to_2}")

# 获取所有行，但只获取第 1 到 2 列 (索引为 0 到 1)
cols_0_to_1 = data_tensor[:, 0:2]
print(f"Cols 0 to 1: \n{cols_0_to_1}")

# 获取第 0 行到第 2 行 (不包括 2 行)，所有列
rows_0_to_1_all_cols = data_tensor[0:2, :]
print(f"Rows 0 to 1 (all cols): \n{rows_0_to_1_all_cols}")

# 3. 布尔索引 (Boolean Indexing)：
print("\n--- Boolean Indexing ---")
rand_tensor_bool = torch.randint(0, 11, (3, 3))
print(f"Original random tensor for boolean indexing:\n{rand_tensor_bool}")

# 找出所有大于 5 的元素
greater_than_5 = rand_tensor_bool[rand_tensor_bool > 5]
print(f"Elements greater than 5: {greater_than_5}")

# 将所有小于 3 的元素设置为 0 (直接修改原张量)
rand_tensor_bool[rand_tensor_bool < 3] = 0
print(f"Tensor after setting elements less than 3 to 0:\n{rand_tensor_bool}")

# 4. 花式索引 (Fancy Indexing)：
print("\n--- Fancy Indexing ---")
fancy_tensor = torch.arange(0, 25).reshape(5, 5)
print(f"Original tensor for fancy indexing:\n{fancy_tensor}")

# 获取第 0, 2, 4 行
selected_rows = fancy_tensor[[0, 2, 4]]
print(f"Selected rows (0, 2, 4):\n{selected_rows}")

# 获取 (0, 0), (1, 2), (2, 4) 这三个特定位置的元素
# 注意：这会返回一个1维张量
specific_elements = fancy_tensor[[0, 1, 2], [0, 2, 4]]
print(f"Specific elements at (0,0), (1,2), (2,4): {specific_elements}")