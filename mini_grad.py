#! /usr/bin/env python
import torch

x = torch.tensor([3.], requires_grad=True)
y = x**2
grad, = torch.autograd.grad(y.mean(), x, create_graph=True)
grad.mean().backward()
print(grad)
print(x.grad)
y = x**2
y.mean().backward()
print(x.grad)

# grad.backward()
# print(grad.grad)
