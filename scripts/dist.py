import argparse

import torch
from tensorboardX import SummaryWriter
import itertools
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', required=True)
parser.add_argument('-lr', type=float, required=True)
args = parser.parse_args()

N_INPUTS = 5
network = torch.nn.Linear(N_INPUTS, 2)
softplus = torch.nn.Softplus()
inputs = torch.ones(1, N_INPUTS)
optimizer = torch.optim.SGD(network.parameters(), lr=args.lr)
log_dir = sys.argv[1]
writer = SummaryWriter(log_dir=args.logdir)

for i in itertools.count():
    a, b = torch.chunk(network(inputs), 2, dim=-1)
    dist = torch.distributions.Normal(a, softplus(b))
    sample = dist.sample()
    j = torch.norm(sample, dim=-1)
    log_prob = dist.log_prob(sample)
    loss = log_prob * j
    writer.add_scalar('loss', loss, i)
    writer.add_scalar('log_prob', log_prob, i)
    writer.add_scalar('J', j, i)
    if i % 100 == 0:
        print(f'loss: {loss}\tJ: {j}')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


