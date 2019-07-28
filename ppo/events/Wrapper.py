import functools
from collections import namedtuple, defaultdict
import numpy as np

import gym

from ppo.events.subtasks import (
    AnswerDoor,
    CatchMouse,
    ComfortBaby,
    MakeDinner,
    MakeFire,
    KillFlies,
    CleanMess,
    AvoidDog,
    WatchBaby,
    Subtask,
)

Obs = namedtuple("Obs", "base subtasks")


class EventsWrapper(gym.Wrapper):
    def __init__(
        self, env, watch_baby_range, avoid_dog_range, door_time_limit, n_active_subtasks
    ):
        super().__init__(env)
        self.n_active_subtasks = n_active_subtasks
        self.possible_subtasks = [
            functools.partial(AnswerDoor, door_time_limit),
            CatchMouse,
            ComfortBaby,
            MakeDinner,
            MakeFire,
            KillFlies,
            CleanMess,
            functools.partial(AvoidDog, avoid_dog_range),
            functools.partial(WatchBaby, watch_baby_range),
        ]
        self.random = env.np_random  # type: np.random
        self.active_mask = None
        self.active_subtasks = None
        env = env.unwrapped
        self.object_one_hots = np.eye(env.height * env.width)

    def reset(self, **kwargs):
        self.active_mask = self.random.randint(2, size=self.n_active_subtasks)
        self.active_subtasks = [
            s for s, a in zip(self.possible_subtasks, self.active_mask) if bool(a)
        ]
        return self.get_observation()

    def step(self, action):
        s, r, t, i = super().step(action)
        object_dict = defaultdict(list)
        for obj in s.objects:
            k = obj.__class__.__name__.lower()
            object_dict[k] += obj
        object_dict = {k: v[0] if len(v) == 1 else v for k, v in object_dict.items()}
        subtask: Subtask
        for subtask in self.active_subtasks:
            subtask.step(*s.interactions, **object_dict)
            if subtask.condition(*s.interactions, **object_dict):
                r += subtask.reward_delta

        return self.get_observation(), r, t, i

    def get_observation(self):
        env = self.env.unwrapped
        dims = env.height, env.width
        base = np.stack(
            [
                self.object_one_hots[np.ravel_multi_index(obj.pos, dims)].reshape(dims)
                * (1 if obj.activated else -1)
                for obj in env.objects
            ]
        )
        return Obs(base=base, subtasks=self.active_mask)
