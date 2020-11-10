from typing import List

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical


class IndexedCategorical(Categorical):
    def __init__(self, choices: torch.tensor, dist: Categorical):
        self.choices = choices.long()

        probs = dist.probs  # n, ..., p
        n, *shape, p = probs.shape
        m = max(choices.max() + 1, p)

        deterministic = torch.eye(m, device=choices.device)  # m, m
        deterministic = deterministic[n:]  # m-n, m
        ones = [1 for _ in shape]  # 1...
        deterministic = deterministic.view(m - n, *ones, m)  # m-n, 1..., m
        deterministic = deterministic.expand(-1, *shape, -1)  # m-n, ..., m

        # More efficient:
        in_range = choices < n
        probs_choices = probs[choices * in_range]
        probs_choices = F.pad(probs_choices, (0, m - p))
        deterministic_choices = deterministic[(choices - n) * ~in_range]
        in_range = in_range.view(-1, *ones, 1)
        chosen = (in_range * probs_choices) + (~in_range * deterministic_choices)

        # Less efficient:
        # padded = F.pad(probs, (0, m - p))  # n, ..., m
        # _chosen = torch.cat([padded, deterministic], dim=0)[choices]  # m, ..., m
        # assert torch.all(chosen == _chosen)

        super().__init__(probs=chosen)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError


class ConditionalDistribution(Distribution):
    def __init__(self, choice: Categorical, options: Categorical, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.choice = choice
        self.options = options

    def sample(self, sample_shape=torch.Size()):
        choice = self.choice.sample()
        chosen = IndexedCategorical(choice, self.options).sample()
        return choice, chosen

    def log_prob(self, choices, *chosen):
        return self.choice.log_prob(choices) + IndexedCategorical(
            choices, self.options
        ).log_prob(chosen)

    def entropy(self):
        return self.choice.entropy() + self.options.entropy().sum(0)

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    @property
    def arg_constraints(self):
        raise NotImplementedError

    @property
    def support(self):
        raise NotImplementedError

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError
