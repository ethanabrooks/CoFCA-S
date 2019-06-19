import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo.teacher import SubtasksTeacher
from ppo.utils import broadcast3d


def g_binary_to_123(g_binary, subtask_space):
    g123 = g_binary.nonzero()[:, 1:].view(-1, 3)
    g123 -= F.pad(
        torch.cumsum(subtask_space, dim=0)[:2], [1, 0], 'constant', 0)
    return g123


class SubtasksStudent(SubtasksTeacher):
    def __init__(self, embedding_dim, tau_diss, tau_diff, task_space, xi,
                 **kwargs):
        self.xi = xi
        self.tau_diss = tau_diss
        self.tau_diff = tau_diff
        self.embedding_dim = embedding_dim
        n_types, max_count, n_objects = task_space.nvec[0]
        super().__init__(**kwargs, task_space=task_space)
        self.embeddings = nn.EmbeddingBag(
            int(task_space.nvec[0].sum()), embedding_dim)
        self.register_buffer(
            'actions',
            torch.cartesian_prod(
                torch.arange(n_types), torch.arange(max_count)).long())
        # TODO: deal with visit when max count > 1
        self.register_buffer('objects',
                             torch.arange(n_objects).long().unsqueeze(1))
        self.register_buffer('subtask_space',
                             torch.tensor(task_space.nvec[0].astype(np.int64)))

    @property
    def d(self):
        return self.obs_sections.base + self.embedding_dim

    def preprocess_obs(self, inputs):
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=1)
        idxs = g_binary_to_123(subtasks[:, :, 0, 0],
                               self.subtask_space).cumsum(dim=-1)
        embedded = self.embeddings(idxs)
        broadcast = broadcast3d(embedded, self.obs_shape[-2:])
        return torch.cat([obs, broadcast], dim=1)

    def forward(self, inputs, *args, action=None, **kwargs):
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=1)
        subtasks = subtasks[:, :, 0, 0]

        g123 = g_binary_to_123(subtasks, self.subtask_space)
        action1, object1 = torch.split(g123, [2, 1], dim=-1)

        def sample_analogy_counterparts(options, exclude):
            excluded = torch.all(
                options == exclude.unsqueeze(1),
                dim=-1,
            )
            n_options = len(options) - 1
            options2 = options[torch.nonzero(1 - excluded)[:, 1]].view(
                obs.size(0), n_options, -1)

            n = obs.size(0)
            idxs = torch.multinomial(
                torch.ones(n_options), num_samples=n, replacement=True).long()
            return options2[torch.arange(n), idxs]

        action2 = sample_analogy_counterparts(self.actions, exclude=action1)
        object2 = sample_analogy_counterparts(self.objects, exclude=object1)

        def embed(action, object):
            idxs = torch.cat([action, object], dim=-1).cumsum(dim=-1)
            return self.embeddings(idxs)

        embedding1 = embed(action1, object2)
        embedding2 = embed(action1, object1)
        embedding3 = embed(action2, object2)
        embedding4 = embed(action2, object1)

        analogy_loss = 0
        for a, b, c, d in [[embedding1, embedding2, embedding3, embedding4],
                           [embedding2, embedding4, embedding1, embedding3]]:
            """
            a-b 1-2 2-4
            | | | | | |
            c-d 3-4 1-3
            """
            sim_loss = F.mse_loss(a - b, c - d)  # a:b::c:d
            dis_loss = -torch.clamp(
                torch.norm(a - d, dim=-1),
                max=self.tau_diss)  # increase the diagonal
            dif_loss = -torch.clamp(
                torch.norm(a - b, dim=-1),
                max=self.tau_diff)  # increase the side
            analogy_loss += (sim_loss + dis_loss + dif_loss)

        act = super().forward(inputs, *args, action=action, **kwargs)
        return act._replace(
            aux_loss=act.aux_loss + self.xi * analogy_loss.mean())
