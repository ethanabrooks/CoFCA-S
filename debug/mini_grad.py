#! /usr/bin/env python
import torch
from torch import optim


def f(x, y):
    return (x + y)**2


def g(x, y):
    return x * y


x = torch.tensor([3.], requires_grad=True)
y = torch.tensor([4.], requires_grad=True)

x_optim = optim.SGD([x], lr=1.)
y_optim = optim.SGD([y], lr=1.)

ddx, = torch.autograd.grad(f(x, y).mean(), x, create_graph=True)
print('d/dx (should be 2(x + y) = 14)', ddx)
ddx.mean().backward()
print('d^2/dx^2 f(x, y) (should be 2):', x.grad)
print('d/dy d/dx f(x, y) (should be 2):', y.grad)

ddx, = torch.autograd.grad(g(x, y).mean(), x)
x.grad = ddx
print('x.grad: (should be d/dx g(x, y) = y = 4)', x.grad)
y_optim.step()
print('value for y after y_optim.step:', y)
print('value for x after y_optim.step:', x)
x_optim.step()
print('value for y after x_optim.step (should be same):', y)
print('value for x after x_optim.step (should be different):', x)
