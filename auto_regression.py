#! /usr/bin/env python

import torch

T, N, D = 10, 2, 3
threshold = torch.tensor([0, 6, -1])  # -1 is a trivial threshold
# inf is how we ensure that finished actions go back to i=0
R = torch.arange(N)
S = torch.arange(D)

A = -1 * torch.ones(T + 1, N, D).long()

samples = torch.tensor(
    [
        [0, 1],
        [1, 4],
        [0, 1],
        [1, 7],
        [1, 7],
        [1, 1],
    ]
)
i = -1

for t, sample in enumerate(samples):
    above_threshold = A[t - 1] > threshold  # meets condition to progress to next action
    sampled = A[t - 1] >= 0  # sampled on a previous time step
    above_threshold[~sampled] = True  # ignore unsampled
    # assert torch.all(sampled.sum(-1) == i + 1)
    above_thresholds = above_threshold.prod(-1)  # met all thresholds
    next_i = sampled.sum(-1) % D  # next i if all thresholds are met
    i: torch.Tensor = above_thresholds * next_i  # otherwise go back to 0
    copy = S.unsqueeze(0) < i.unsqueeze(1)  # actions accumulated from prev time steps
    A[t][copy] = A[t - 1][copy]  # copy accumulated actions from A[t-1]
    A[t, R, i] = sample  # set i^th column to new action
    print("sample")
    print(sample)
    print(f"A[{t}]")
    print(A[t])


if __name__ == "__main__":
    pass
