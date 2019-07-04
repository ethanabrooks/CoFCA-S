import torch
import torch.nn.functional as F
import ppo.subtasks

import ppo.control_flow


class Agent(ppo.control_flow.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.control_flow.agent.Recurrence):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size, **kwargs)
        self.register_buffer(
            "true_path", F.pad(torch.eye(self.n_subtasks), [1, 0])[:, :-1]
        )
        self.register_buffer(
            "false_path", F.pad(torch.eye(self.n_subtasks), [2, 0])[:, :-2]
        )

    def inner_loop(self, M, inputs, **kwargs):
        def update_attention(p, t):
            r = (p.unsqueeze(1) @ M).squeeze(1)
            pred = self.phi_shift((inputs.base[t], r))  # TODO
            trans = pred * self.true_path + (1 - pred) * self.false_path
            return (p.unsqueeze(1) @ trans).squeeze(1)

        kwargs.update(update_attention=update_attention)
        yield from ppo.subtasks.agent.Recurrence.inner_loop(
            inputs=inputs, M=M, **kwargs
        )
