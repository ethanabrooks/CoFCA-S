from enum import Enum

import torch
from utils import ReplayBuffer, onehot

from common.running_mean_std import RunningMeanStd
from ppo.util import Categorical, NoInput, init_normc_, mlp

SamplingStrategy = Enum('SamplingStrategy',
                        'baseline pg gpg l2g gl2g abs_grads')


class TaskGenerator(NoInput):
    def __init__(self, task_size, temperature: float, exploration_bonus: float,
                 sampling_strategy):
        super().__init__(task_size)
        self.exploration_bonus = exploration_bonus
        self.sampling_strategy = sampling_strategy
        self.task_size = task_size
        self.temperature = temperature
        if sampling_strategy == SamplingStrategy.abs_grads.name:
            self._logits = torch.Tensor(1, task_size)
            init_normc_(self._logits)
            self._logits = self._logits.view(-1)
        else:
            self._logits = torch.ones(task_size)

    @property
    def logits(self):
        return self._logits

    def dist(self):
        return Categorical(logits=self.temperature * self.logits)

    def sample(self):
        return self.dist().sample()

    def probs(self):
        return self.dist().probs.detach().numpy()

    def importance_weight(self, probs):
        return 1 / (self.task_size * probs)

    def update(self, tasks, grads, step=None):
        if self.sampling_strategy != 'baseline':
            self._logits += self.exploration_bonus
            self._logits[tasks] = grads.cpu()


class RewardBasedTaskGenerator(TaskGenerator):
    def __init__(self, task_size, **kwargs):
        super().__init__(task_size=task_size, **kwargs)
        self.histories = [ReplayBuffer(100) for _ in task_size]

    @property
    def logits(self):
        logits = [h.array().var() for h in self.histories]
        return torch.tensor(logits)

    def update(self, tasks, rewards, step=None):
        for task, reward in zip(tasks, rewards):
            self.histories[task].append(reward)


class GoalGAN(RewardBasedTaskGenerator):
    def __init__(self, size_noise, network_args, reward_lower_bound, reward_upper_bound,
                 lrG, lrD, writer,
                 **kwargs):
        super().__init__(**kwargs)
        self.reward_lower_bound = reward_lower_bound
        self.reward_upper_bound = reward_upper_bound
        self.noise_dist = torch.distributions.Normal(loc=torch.zeros(size_noise),
                                                     scale=torch.ones(size_noise))
        self.G = mlp(num_inputs=size_noise, num_outputs=1, **network_args)
        self.D = mlp(num_inputs=size_noise, num_outputs=1, **network_args)
        self.noise = None
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lrG)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=lrD)
        self.writer = writer

    def sample(self):
        self.noise = self.noise_dist.sample()
        return self.G(self.noise)

    def update(self, tasks, rewards, step=None):
        super().update(tasks, rewards)
        task_vectors = onehot(tasks, self.num_tasks)
        task_vectors = torch.tensor(task_vectors)
        y = (rewards > self.reward_lower_bound) & (rewards < self.reward_upper_bound)
        loss_d = y * (self.D(task_vectors) - 1)**2 + (1-y) * (self.D(task_vectors) + 1)**2
        loss_d.mean().backward(retain_graph=True)
        self.D_optimizer.step()
        self.D_optimizer.zero_grad()
        loss_g = self.D(self.G(self.noise_dist.sample()))**2
        loss_g.mean().backward()
        self.G_optimizer.step()
        self.G_optimizer.zero_grad()
        if self.writer:
            self.writer.add_scalar('discriminator loss', loss_d, step)
            self.writer.add_scalar('generator loss', loss_g, step)

