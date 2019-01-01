import torch

x = torch.ones(1, requires_grad=True)
net = torch.nn.Linear(1, 1)

# optim_x = torch.optim.Adam([x], lr=1)
# optim_net = torch.optim.Adam(net.parameters(), lr=1)

y = net(x)
grads = torch.autograd.grad(y, net.parameters(), create_graph=True)
norms = [grad.norm(2) for grad in grads]
norm = torch.stack(norms)
norm = torch.sum(norm)

norm.backward()

# grad = sum(grads, torch.zeros((), requires_grad=True))
# import ipdb; ipdb.set_trace()
print(x.grad)
for p in net.parameters():
    print(p.grad)
y.backward()
# optim_net.zero_grad()
for p in net.parameters():
    print(p.grad)
