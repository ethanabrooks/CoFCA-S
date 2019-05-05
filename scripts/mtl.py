import torch

from ppo.util import frank_wolfe_solver

if __name__ == '__main__':
    gradients = torch.tensor([[-4, 1], [2, 1]]).float()
    alphas = frank_wolfe_solver(gradients, 3)
    print(alphas @ gradients)
