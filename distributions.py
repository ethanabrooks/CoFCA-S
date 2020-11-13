import torch
from torch.distributions import Distribution, Categorical


def unravel_index(indices: torch.Tensor, *shape: int):
    def unravel(ind):
        for dim in reversed(shape):
            yield ind % dim
            ind = indices // dim

    return tuple(reversed(list(unravel(indices))))


def ravel_multi_index(
    multi_index: torch.Tensor, *dims: int, validate=False
) -> torch.Tensor:
    n, *shape = multi_index.shape

    if validate:
        dims = torch.tensor(dims, device=multi_index.device)
        # noinspection PyTypeChecker
        assert torch.all(multi_index < dims.view(n, *(1 for _ in shape)))

    _, *dims = dims
    dims = torch.tensor([*dims, 1], device=multi_index.device)
    dims = torch.cumprod(dims.flip(0), 0).flip(0)
    dims = dims.view(n, *(1 for _ in shape))
    return torch.sum(dims * multi_index, dim=0)


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
