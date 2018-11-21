#! /usr/bin/env python

import torch
a = torch.tensor([1., 2.])
x = torch.tensor([0., 0.], requires_grad=True)
print('x', x)
x.detach()[0] = a[1]
print('x', x)
y = torch.tensor(3., requires_grad=True)
z = (0.5 * x ** 2).sum()
w = 4 * y

z.backward()
print('x.grad', x.grad)
print('y.grad', y.grad)
# y.grad = None
print('y.grad', y.grad)
w.backward()
print('y.grad', y.grad)
w.backward()
import ipdb
ipdb.set_trace()
