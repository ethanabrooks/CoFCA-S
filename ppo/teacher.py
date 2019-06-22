import torch
from torch.nn import functional as F

from ppo.agent import Agent
from ppo.wrappers import SubtasksActions, get_subtasks_obs_sections


class SubtasksTeacher(Agent):
    def __init__(self, task_space, obs_space, action_space, **kwargs):
        self.obs_sections = get_subtasks_obs_sections(task_space)
        d, h, w = self.obs_shape = obs_space.shape
        assert d == sum(self.obs_sections)
        self.action_spaces = SubtasksActions(*action_space.spaces)
        super().__init__(
            obs_shape=(self.d, h, w),
            action_space=self.action_spaces.a,
            **kwargs)

    @property
    def d(self):
        return self.obs_sections.base + self.obs_sections.subtask

    def preprocess_obs(self, inputs):
        obs, subtasks, task_broad, next_subtask_broad = torch.split(
            inputs, self.obs_sections, dim=1)
        return torch.cat([obs, subtasks], dim=1)

    def forward(self, inputs, *args, action=None, **kwargs):
        if action is not None:
            action = action[:, :1]

        act = super().forward(
            self.preprocess_obs(inputs), action=action, *args, **kwargs)
        x = torch.zeros_like(act.action)
        actions = SubtasksActions(a=act.action, g=x, cg=x, cr=x)
        return act._replace(action=torch.cat(actions, dim=-1))

    def get_value(self, inputs, rnn_hxs, masks):
        return super().get_value(self.preprocess_obs(inputs), rnn_hxs, masks)


def g_binary_to_123(g_binary, subtask_space):
    g123 = g_binary.nonzero()[:, 1:].view(-1, 3)
    g123 -= F.pad(
        torch.cumsum(subtask_space, dim=0)[:2], [1, 0], 'constant', 0)
    return g123
