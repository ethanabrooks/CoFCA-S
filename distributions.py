# third party
import torch
import torch.nn as nn

# first party
from torch.distributions import Distribution

from utils import AddBias, init, init_normc_

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)


# log_prob_cat = FixedCategorical.log_prob
def log_prob_cat(self, value):
    if self._validate_args:
        self._validate_sample(value)
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, self.logits)
    value = value[..., :1]
    # gather = log_pmf.gather(-1, value).squeeze(-1)
    R = torch.arange(value.size(0))
    return log_pmf[R, value.squeeze(-1)]  # deterministic


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


class JointCategorical(Categorical):
    def __init__(
        self, distribution: Categorical, *distributions: Categorical, **kwargs
    ):
        *shape, _ = distribution.probs.shape
        probs = distribution.probs.unsqueeze(-1)
        for i, distribution in enumerate(distributions):
            probs = probs * distribution.probs.unsqueeze(-2)
            probs = probs.view(*shape, -1)
        super().__init__(probs=probs, **kwargs)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError


class JointDistribution(Distribution):
    def __init__(self, *distributions: Distribution):
        super().__init__()
        self.distributions = distributions

    def sample(self, sample_shape=torch.Size()):
        return [d.sample() for d in self.distributions]

    def log_probs(self, *values):
        return sum([d.log_probs(v) for d, v in zip(self.distributions, values)])

    def log_prob(self, choices, *chosen):
        raise NotImplementedError

    def entropy(self):
        return sum(d.entropy() for d in self.distributions)

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
