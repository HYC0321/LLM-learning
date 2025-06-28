import torch

x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3.)
z = torch.tensor(4., requires_grad=True)

a = x * y
b = a + z
c = b.sum()

print(f"a.grad_fn: {a.grad_fn}")
print(f"b.grad_fn: {b.grad_fn}")
print(f"c.grad_fn: {c.grad_fn}")

print(f"x.requires_grad: {x.requires_grad}")
print(f"y.requires_grad: {y.requires_grad}")
print(f"z.requires_grad: {z.requires_grad}")
print(f"a.requires_grad: {a.requires_grad}")
print(f"b.requires_grad: {b.requires_grad}")
print(f"c.requires_grad: {c.requires_grad}")

print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")
print(f"z.grad: {z.grad}")
c.backward()
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")
print(f"z.grad: {z.grad}")