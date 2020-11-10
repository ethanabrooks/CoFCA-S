import torch
from torch import nn as nn

from utils import init, init_normc_, AddBias

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical
old_sample = FixedCategorical.sample
log_prob_cat = FixedCategorical.log_prob
FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
entropy = FixedNormal.entropy

FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)
).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(
    -1, keepdim=True
)

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


class AutoRegressive(nn.Module):
    def __init__(self, num_inputs, num_choices, num_outputs):
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )
        self.num_choices = num_choices
        self.choices_logits = init_(nn.Linear(num_inputs, num_choices))
        self.chosen = init_(nn.Linear(num_inputs, num_choices * num_outputs))

    def forward(self, x):
        N = x.size(0)
        R = torch.arange(N, device=x.device)
        choices = self.choices_logits(x)
        chosen = self.chosen(x).view(N, self.num_choices, -1)[R, choices]
