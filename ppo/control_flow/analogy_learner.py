import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo.control_flow.lower_level import LowerLevel


class AnalogyLearner(LowerLevel):
    def __init__(self, obs_spaces, embedding_dim, tau_diss, tau_diff, xi, **kwargs):
        self.xi = xi
        self.tau_diss = tau_diss
        self.tau_diff = tau_diff
        self.embedding_dim = embedding_dim
        subtask_nvec = obs_spaces.subtask.nvec
        n_types, max_count, n_objects = subtask_nvec
        super().__init__(**kwargs, obs_spaces=obs_spaces)
        self.embeddings = nn.EmbeddingBag(int(subtask_nvec.sum()), embedding_dim)

        self.register_buffer(
            "actions",
            torch.cartesian_prod(torch.arange(n_types), torch.arange(max_count)).long(),
        )
        # TODO: deal with visit when max count > 1
        self.register_buffer("objects", torch.arange(n_objects).long().unsqueeze(1))
        self.register_buffer(
            "subtask_space", torch.tensor(subtask_nvec.astype(np.int64))
        )

    @property
    def d(self):
        return self.obs_spaces.base.shape[0] * self.embedding_dim
        # return self.obs_sections.base + self.embedding_dim

    def preprocess_obs(self, inputs):
        obs, g123 = torch.split(inputs, self.obs_sections, dim=1)[:2]
        n = obs.size(0)
        obs = obs.view(n, *self.obs_shape).unsqueeze(1)
        embedding = self.embeddings(g123.cumsum(dim=-1).long()).view(
            n, self.embedding_dim, 1, 1, 1
        )
        return (obs * embedding).view(n, self.d, *self.obs_shape[-2:])

    def forward(self, inputs, *args, action=None, **kwargs):
        obs, g123 = torch.split(inputs, self.obs_sections, dim=1)[:2]
        obs = obs.view(obs.size(0), *self.obs_shape)
        action1, object1 = torch.split(g123.long(), [2, 1], dim=-1)
        n = obs.size(0)

        def sample_analogy_counterparts(options, exclude):
            exclude = exclude.view(n, 1, -1, options.shape[1])
            m, d = options.shape
            # noinspection PyTypeChecker
            excluded = torch.any(
                torch.all(exclude == options.view(1, m, 1, d), dim=-1), dim=-1
            )

            def sample():
                for e in excluded:
                    o = options[e == 0]
                    yield o[np.random.randint(low=0, high=len(o))]

            return torch.stack(list(sample()))

        action2 = sample_analogy_counterparts(
            self.actions, exclude=action1.unsqueeze(1)
        )
        object2 = sample_analogy_counterparts(
            self.objects, exclude=object1.unsqueeze(1)
        )
        action3 = sample_analogy_counterparts(
            self.actions, exclude=torch.stack([action1, action2], dim=1)
        )
        object3 = sample_analogy_counterparts(
            self.actions, exclude=torch.stack([object1, object2], dim=1)
        )

        def embed(*values):
            return self.embeddings(torch.cat(values, dim=-1).cumsum(dim=-1))

        embedding1 = embed(action1, object1)
        embedding2 = embed(action1, object2)
        embedding3 = embed(action2, object1)
        embedding4 = embed(action2, object2)
        embedding5 = embed(action3, object2)
        embedding6 = embed(action1, object3)

        sim_loss = 0
        dis_loss = 0
        dif_loss = 0
        for a, b, c, d in [
            [embedding1, embedding2, embedding3, embedding4],
            [embedding1, embedding3, embedding2, embedding4],
        ]:
            """
            a-b 1-2 2-4
            | | | | | |
            c-d 3-4 1-3
            """
            sim_loss += F.mse_loss(a - b, c - d)  # a:b::c:d
            dif_loss += -torch.clamp(
                torch.norm(a - b, dim=-1), max=self.tau_diff
            ).mean()  # increase the side

        for a, b, c, d in [
            [embedding1, embedding2, embedding5, embedding4],
            [embedding1, embedding2, embedding3, embedding6],
        ]:
            dis_loss -= torch.clamp(F.mse_loss(a - b, c - d), max=self.tau_diss)

        analogy_loss = sim_loss + dis_loss + dif_loss
        act = super().forward(inputs, *args, action=action, **kwargs)
        act.log.update(
            sim_loss=sim_loss,
            dis_loss=dis_loss,
            dif_loss=dif_loss,
            analogy_loss=analogy_loss,
        )
        return act._replace(aux_loss=act.aux_loss + self.xi * analogy_loss)
