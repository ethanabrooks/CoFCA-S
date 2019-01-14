import torch


x = torch.tensor(1., requires_grad=True)
y = torch.tensor(1., requires_grad=True)
z = (x * y) ** 2
grads = torch.autograd.grad(outputs=z,
                            inputs=[x],
                            create_graph=True)
grad = sum(grads)
grad.backward()


