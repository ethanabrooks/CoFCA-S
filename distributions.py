import torch
import torch.nn.functional as F
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


class IndexedCategorical(Categorical):
    def __init__(self, choices: torch.tensor, dist: Distribution):
        self.choices = choices.long()
        probs = dist.probs  # n, ..., p
        super().__init__(probs=self.indexed_probs(choices, probs))

    @staticmethod
    def indexed_probs(choices: torch.Tensor, probs: torch.Tensor):
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
        probs_choices = F.pad(probs_choices, (0, m - p))  # this will clip if m < p
        deterministic_choices = deterministic[(choices - n) * ~in_range]
        # noinspection PyUnresolvedReferences
        in_range = in_range.view(-1, *ones, 1)
        chosen = (in_range * probs_choices) + (~in_range * deterministic_choices)

        # Less efficient:
        # padded = F.pad(probs, (0, m - p))  # n, ..., m
        # _chosen = torch.cat([padded, deterministic], dim=0)[choices]  # m, ..., m
        # assert torch.all(chosen == _chosen)
        return chosen

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError


class BatchIndexedCategorical(IndexedCategorical):
    def indexed_probs(self, choices: torch.Tensor, probs: torch.Tensor):
        assert choices.ndimension() == 1
        arange = torch.arange(choices.numel())
        return super().indexed_probs(choices, probs)[arange, arange]

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError


class ExpandCategorical(Categorical):
    def __init__(self, distribution: Categorical, shape: torch.Size):
        super().__init__(probs=distribution.probs.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError


class UnsqueezeCategorical(Categorical):
    def __init__(self, distribution: Categorical, index: int):
        super().__init__(probs=distribution.probs.unsqueeze(index))

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError


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


class ConditionalCategorical(Categorical):
    def __init__(
        self, condition: Categorical, conditioned: Categorical, *args, **kwargs
    ):
        *shape, m, n = conditioned.probs.shape
        *shape2, o = condition.probs.shape
        assert tuple(shape) == tuple(shape2)
        p = max(n, o)

        probs = F.pad(conditioned.probs, (p - n, 0))  # ..., m, p

        if o > m:
            deterministic = torch.eye(p, device=condition.probs.device)  # p, p
            deterministic = deterministic[: o - m]  # o-m, p
            ones = [1 for _ in shape]  # 1...
            deterministic = deterministic.view(*ones, o - m, n)  # 1..., o-m, p
            deterministic = deterministic.expand(*shape, -1, -1)  # ..., o-m, p
            probs = torch.cat([deterministic, probs], dim=-2)  # ..., o, p

        condition_probs = condition.probs.unsqueeze(-1)  # 1..., o, 1
        conditional_probs = condition_probs * probs  # ..., o, p
        probs = conditional_probs.view(*shape, -1)  # ..., o * p
        self.p = p
        super().__init__(probs=probs, *args, **kwargs)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError


class BatchConditionalCategorical(ConditionalCategorical):
    def indexed_categorical(self, choice):
        return BatchIndexedCategorical(choice, self.options)

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
