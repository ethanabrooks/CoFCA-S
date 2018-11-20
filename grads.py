#! /usr/bin/env python

import torch
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = x + y
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
