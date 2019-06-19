import torch
import torch.nn as nn

from ppo.agent import Agent
from ppo.teacher import SubtasksTeacher
from ppo.utils import broadcast3d
from ppo.wrappers import SubtasksActions, get_subtasks_obs_sections


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
