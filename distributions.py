# third party
from typing import List
from torch.distributions import Distribution

import torch
import torch.nn as nn

# first party
from utils import AddBias, init, init_normc_

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)
).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(
    -1, keepdim=True
)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros_like(action_mean)
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


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


class AutoRegressive(Distribution):
    def __init__(
        self, logits: torch.Tensor, children: List[Distribution], *args, **kwargs
    ):
        assert logits.size(-1) == len(children)
        self.children = children
        self.parent = FixedCategorical(logits=logits)
        super().__init__(*args, **kwargs)

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
        choice = self.parent.sample(sample_shape)
        import ipdb

        ipdb.set_trace()
        cat = torch.cat([choice, *self.children[choice].sample(sample_shape)])
        return cat

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, value):
        head, *tail = value
        return self.parent.log_prob(head) + self.children[head].log_prob(*tail)

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def entropy(self):
        import ipdb

        ipdb.set_trace()
        return self.parent.entropy() + self.parent.probs * torch.stack(
            [d.entropy() for d in self.children]
        )
