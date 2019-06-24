import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo.subtasks.teacher import Teacher, g_binary_to_123


class Student(Teacher):
    def __init__(self, embedding_dim, tau_diss, tau_diff, task_space, xi,
                 **kwargs):
        self.xi = xi
        self.tau_diss = tau_diss
        self.tau_diff = tau_diff
        self.embedding_dim = embedding_dim
        self.task_space = task_space
        n_types, max_count, n_objects = task_space.nvec[0]
        super().__init__(**kwargs, task_space=task_space)
        # self.embeddings = nn.EmbeddingBag(
        #     int(task_space.nvec[0].sum()), embedding_dim)
        self.embeddings = nn.ModuleList(
            nn.Embedding(n, embedding_dim) for n in task_space.nvec[0])

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
        return self.obs_sections.base * self.embedding_dim**3
        # return self.obs_sections.base + self.embedding_dim

    def preprocess_obs(self, inputs):
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=1)
        n, d, h, w = obs.shape
        g123 = g_binary_to_123(subtasks[:, :, 0, 0], self.subtask_space)
        g123 = torch.split(g123, [1, 1, 1], dim=-1)
        embeddings = [e(s.flatten()) for e, s in zip(self.embeddings, g123)]
        obs6d = 1
        for i1, part in enumerate(embeddings):
            for i2 in range(len(embeddings) + 2):  # 2 for h,w
                if i1 != i2:
                    part.unsqueeze_(i2 + 1)
            obs6d = obs6d * part
        obs7d = obs.view(n, d, 1, 1, 1, h, w) * obs6d.unsqueeze(1)
        return obs7d.view(n, -1, h, w)
        # broadcast = broadcast3d(embedded, self.obs_shape[-2:])
        # return torch.cat([obs, broadcast], dim=1)

        # idxs = g_binary_to_123(subtasks[:, :, 0, 0],
        #                        self.subtask_space).cumsum(dim=-1)
        # embedded = self.embeddings(idxs)
        # broadcast = broadcast3d(embedded, self.obs_shape[-2:])
        # return torch.cat([obs, broadcast], dim=1)

    def forward(self, inputs, *args, action=None, **kwargs):
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=1)
        n = obs.size(0)
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

            idxs = torch.multinomial(
                torch.ones(n_options), num_samples=n, replacement=True).long()
            return options2[torch.arange(n), idxs]

        action2 = sample_analogy_counterparts(self.actions, exclude=action1)
        object2 = sample_analogy_counterparts(self.objects, exclude=object1)

        def embed(*values):
            values = torch.cat(values, dim=-1)
            embeds = 1
            for i, v in enumerate(torch.split(values, [1, 1, 1], dim=-1)):
                emb = self.embeddings[i](v.flatten())
                for j in range(3):
                    if i != j:
                        emb.unsqueeze_(j + 1)
                embeds = embeds * emb
            return embeds.view(n, -1)
            # idxs = torch.cat([action, object], dim=-1).cumsum(dim=-1)
            # return self.embeddings(idxs)

        embedding1 = embed(action1, object2)
        embedding2 = embed(action1, object1)
        embedding3 = embed(action2, object2)
        embedding4 = embed(action2, object1)

        sim_loss = 0
        dis_loss = 0
        dif_loss = 0
        for a, b, c, d in [[embedding1, embedding2, embedding3, embedding4],
                           [embedding2, embedding4, embedding1, embedding3]]:
            """
            a-b 1-2 2-4
            | | | | | |
            c-d 3-4 1-3
            """
            sim_loss += F.mse_loss(a - b, c - d)  # a:b::c:d
            dis_loss += -torch.clamp(
                torch.norm(a - d, dim=-1),
                max=self.tau_diss).mean()  # increase the diagonal
            dif_loss += -torch.clamp(
                torch.norm(a - b, dim=-1),
                max=self.tau_diff).mean()  # increase the side

        analogy_loss = sim_loss + dis_loss + dif_loss
        act = super().forward(inputs, *args, action=action, **kwargs)
        act.log.update(
            sim_loss=sim_loss,
            dis_loss=dis_loss,
            dif_loss=dif_loss,
            analogy_loss=analogy_loss)
        return act._replace(aux_loss=act.aux_loss + self.xi * analogy_loss)
