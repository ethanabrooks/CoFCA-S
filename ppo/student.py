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
    def __init__(self, embedding_dim, **kwargs):
        self.embedding_dim = embedding_dim
        super().__init__(**kwargs)
        self.embeddings = nn.EmbeddingBag(self.action_spaces.g_int.n,
                                          embedding_dim)

    @property
    def d(self):
        return self.obs_sections.base + self.embedding_dim

    def preprocess_obs(self, inputs):
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=1)
        subtasks = None  # TODO
        embedded = self.embeddings(subtasks).view()  # TODO
        subtasks = broadcast3d(embedded, self.obs_shape[-2:])
        return torch.cat([obs, subtasks], dim=-1)

    def forward(self, inputs, *args, action=None, **kwargs):
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=1)
        subtasks = subtasks[:, :, 0, 0]
        g123 = g_binary_to_123(subtasks, self.subtask_space)
        action1, object1 = torch.split(g123, [2, 1], dim=-1)

        def sample_analogy_counterparts(options, exclude):
            options = options[1 - torch.all(options == exclude)]
            n = obs.size(0)
            idxs = torch.multinomial(
                torch.arange(len(options) - 1),
                num_samples=n,
                replacement=True)
            return options[torch.arange(n), idxs]

        action2 = sample_analogy_counterparts(self.actions, exclude=action1)
        object2 = sample_analogy_counterparts(self.objects, exclude=object1)

        def embed(action, object):
            composed = torch.cat([action, object], dim=-1).cumsum(dim=-1)
            return self.embeddings(composed)

        subtask1 = torch.cat([action1, object2], dim=-1).cumsum(dim=-1)

        embeddings = self.embeddings(torch.nonzero(subtasks))
