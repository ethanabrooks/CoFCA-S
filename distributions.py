from typing import List

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical


class IndexedCategorical(Categorical):
    def __init__(self, choices: torch.tensor, probs, *args, **kwargs):
        self.choices = choices.long()
        dist = Categorical(probs, *args, **kwargs)
        probs = dist.probs  # n, ..., p
        n, *shape, p = probs.shape
        m = max(choices.max() + 1, p)
        padded = F.pad(probs, (0, m - p))  # n, ..., m
        deterministic = torch.eye(m, device=choices.device)  # m, m
        deterministic = deterministic[n:]  # m-n, m
        ones = [1 for _ in shape]  # 1...
        deterministic = deterministic.view(m - n, *ones, m)  # m-n, 1..., m
        deterministic = deterministic.expand(-1, *shape, -1)  # m-n, ..., m
        # More efficient:
        in_range = choices < n  # c
        probs_choices = padded[choices * in_range]
        deterministic_choices = deterministic[(choices - n) * ~in_range]
        in_range = in_range.view(-1, *ones, 1)
        chosen = (in_range * probs_choices) + (~in_range * deterministic_choices)
        # Less efficient:
        # _chosen = torch.cat([padded, deterministic], dim=0)[choices]  # m, ..., m
        # assert torch.all(chosen == _chosen)
        super().__init__(probs=chosen, *args, **kwargs)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError


class ConditionalDistribution(Distribution):
    def __init__(
        self, choice: Categorical, distributions: Categorical, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        probs = choice.probs
        self.deterministic = torch.arange(
            distributions.probs.size(0), probs.size(-1), device=probs.device
        )
        self.choice = choice
        self.distributions = distributions

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

    def sample(self, sample_shape=torch.Size()):
        options = self.distributions.sample(sample_shape)
        _, *shape = options.shape
        deterministic = self.deterministic.view(-1, *(1 for _ in shape))
        options = torch.cat([options, deterministic.expand(-1, *shape)], dim=0)
        choices = self.choice.sample()
        return choices, options[choices]

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, choices, *chosen):
        return self.choice.log_prob(choices) + self.deterministic.log_prob()

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError


class JointDistribution(Distribution):
    def __init__(self, distributions: List[Distribution], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distributions = distributions

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

    def sample(self, sample_shape=torch.Size()):
        return torch.cat([d.sample(sample_shape) for d in self.distributions])

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, value):
        import ipdb

        ipdb.set_trace()
        return sum([d.log_prob(v) for d, v in zip(self.distributions, value)])

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def entropy(self):
        return sum([d.entropy() for d in self.distributions])
