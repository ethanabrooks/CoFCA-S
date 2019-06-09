import torch

from ppo.agent import Agent
from ppo.wrappers import get_subtasks_obs_sections, SubtasksActions


class SubtasksTeacher(Agent):
    def __init__(self, task_space, obs_shape, action_space, **kwargs):
        self.obs_sections = get_subtasks_obs_sections(task_space)
        d, h, w = obs_shape
        assert d == sum(self.obs_sections)
        self.d = self.obs_sections.base + self.obs_sections.subtask
        self.action_spaces = SubtasksActions(*action_space.spaces)
        super().__init__(
            obs_shape=(self.d, h, w),
            action_space=self.action_spaces.a,
            **kwargs)

    def preprocess_obs(self, inputs):
        batch_size, _, h, w = inputs.shape
        base_obs = inputs[:, :self.obs_sections.base]
        start = self.obs_sections.base
        stop = self.obs_sections.base + self.obs_sections.subtask
        subtask = inputs[:, start:stop, :, :]
        return torch.cat([base_obs, subtask], dim=1)

    def forward(self, inputs, *args, **kwargs):
        act = super().forward(self.preprocess_obs(inputs), *args, **kwargs)
        n = inputs.shape[0]
        g = torch.zeros_like(act.action)
        b = torch.zeros_like(act.action)
        actions = SubtasksActions(a=act.action, b=b, g=g)
        return act._replace(action=torch.cat(actions, dim=-1))

    def get_value(self, inputs, rnn_hxs, masks):
        return super().get_value(self.preprocess_obs(inputs), rnn_hxs, masks)