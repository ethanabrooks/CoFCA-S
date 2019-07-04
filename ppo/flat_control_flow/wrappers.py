from gym import spaces

import ppo.subtasks


class Wrapper(ppo.subtasks.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space.spaces.update(
            g=spaces.Discrete(len(env.observation_space.spaces["lines"].nvec))
        )

    def render_assigned_subtask(self):
        env = self.env.unwrapped
        g_type, g_count, g_obj, condition = tuple(env.lines[self.last_g])
        if condition:
            print("if", self.object_types[condition - 1])
        else:
            print(
                self.last_g,
                env.interactions[g_type],
                g_count + 1,
                env.object_types[g_obj],
            )
