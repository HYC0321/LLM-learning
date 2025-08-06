import torch

print("--- 练习 1：基础梯度计算 ---")

# 创建张量并设置 requires_grad
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0)  # 默认 requires_grad=False
z = torch.tensor(4.0, requires_grad=True)

print(f"初始张量：x={x}, y={y}, z={z}")

# 执行操作构建计算图
a = x * y
b = a + z
c = b.sum() # 确保 c 是一个标量，这里 b 已经是一个标量，所以 sum() 不改变值，但确保了grad_fn

print(f"中间结果：a={a}, b={b}, c={c}")

# 检查属性
print("\n--- 检查 grad_fn ---")
print(f"a.grad_fn: {a.grad_fn}")
print(f"b.grad_fn: {b.grad_fn}")
print(f"c.grad_fn: {c.grad_fn}")

print("\n--- 检查 requires_grad ---")
print(f"x.requires_grad: {x.requires_grad}")
print(f"y.requires_grad: {y.requires_grad}")
print(f"z.requires_grad: {z.requires_grad}")
print(f"a.requires_grad: {a.requires_grad}")
print(f"b.requires_grad: {b.requires_grad}")
print(f"c.requires_grad: {c.requires_grad}")

# 执行反向传播
print("\n--- 执行反向传播 ---")
c.backward()

# 打印梯度
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")
print(f"z.grad: {z.grad}")

print("\n--- 梯度解释 ---")
print("y.grad 为 None，因为 y.requires_grad 为 False。只有 requires_grad=True 的叶子张量（以及通过它们计算得到但仍需要梯度的中间张量）才会累积梯度。")
print("计算过程：")
print("c = b = a + z = (x * y) + z")
print("dc/dx = d((x * y) + z)/dx = y")
print("dc/dy = d((x * y) + z)/dy = x")
print("dc/dz = d((x * y) + z)/dz = 1")
print(f"所以 x.grad 应该为 y 的值 ({y.item()})，即 {x.grad.item()}")
print(f"z.grad 应该为 1 ({z.grad.item()})")