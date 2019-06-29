#! /usr/bin/env python

import ipdb
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x + y
w = 4 * y

z.backward()
print("x.grad", x.grad)
print("y.grad", y.grad)
# y.grad = None
print("y.grad", y.grad)
w.backward()
print("y.grad", y.grad)
w.backward()
ipdb.set_trace()
