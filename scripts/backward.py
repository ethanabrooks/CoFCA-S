import torch

x = torch.ones(1)
net = torch.nn.Linear(1, 1)

y = net(x)
grads = torch.autograd.grad(y, net.parameters())
y.backward(gradient=grads[0])
