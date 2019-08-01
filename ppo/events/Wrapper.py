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
import ppo.events

Obs = namedtuple("Obs", "base subtasks")


class Wrapper(gym.Wrapper):
    def __init__(
        self, env, watch_baby_range, avoid_dog_range, door_time_limit, n_active_subtasks
    ):
        super().__init__(env)
        self.n_active_subtasks = n_active_subtasks
        self.possible_subtasks = [
            AnswerDoor(door_time_limit),
            CatchMouse(),
            ComfortBaby(),
            MakeDinner(),
            MakeFire(),
            KillFlies(),
            CleanMess(),
            AvoidDog(avoid_dog_range),
            WatchBaby(watch_baby_range),
        ]
        self.active_mask = None
        self.active_subtasks = None
        env = env.unwrapped
        self.width, self.height = env.width, env.height
        self.object_one_hots = np.eye(env.height * env.width)

    def render(self, mode="human", **kwargs):
        grid_height = 1
        env = self.env.unwrapped
        object_string = defaultdict(str)
        object_count = defaultdict(int)
        for obj in env.objects:
            if obj.pos is not None:
                string = "{:^3}".format(obj.icon())
                object_string[obj.pos] += string
                object_count[obj.pos] += len(string)

        width = max(max(object_count.values()), 10)
        object_string = {
            k: "{:^{width}}".format(v, width=width) for k, v in object_string.items()
        }
        grid_width = max(len(s) for s in object_string.values())
        print(max(object_string.values(), key=lambda x: len(x)))
        print(grid_width)

        for i in range(self.height):
            print("\n" + "-" * self.width * (1 + width))
            for j in range(self.width):
                print(object_string.get((i, j), " " * width), end="|")
        print()

    def reset(self, **kwargs):
        self.active_mask = np.random.randint(2, size=self.n_active_subtasks)
        self.active_subtasks = [
            s for s, a in zip(self.possible_subtasks, self.active_mask) if bool(a)
        ]
        return self.observation(super().reset())

    def step(self, action):
        s, r, t, i = super().step(action)
        object_dict = defaultdict(list)
        for obj in s.objects:
            k = obj.__class__.__name__.lower()
            object_dict[k] += [obj]
        object_dict = {k: v[0] if len(v) == 1 else v for k, v in object_dict.items()}
        subtask: Subtask
        for subtask in self.active_subtasks:
            subtask.step(*s.interactions, **object_dict)  # TODO wtf
            if subtask.condition(*s.interactions, **object_dict):
                r += subtask.reward_delta

        return self.observation(s), r, t, i

    def observation(self, observation):
        dims = self.height, self.width
        object_pos = [
            self.object_one_hots[np.ravel_multi_index(obj.pos, dims)].reshape(dims)
            * (-1 if obj.activated else 1)
            for obj in observation.objects
            if obj.pos is not None
        ]
        base = np.stack(object_pos)
        return Obs(base=base, subtasks=self.active_mask)


if __name__ == "__main__":
    import gridworld_env.keyboard_control

    env = Wrapper(
        watch_baby_range=2,
        avoid_dog_range=2,
        door_time_limit=10,
        n_active_subtasks=2,
        env=ppo.events.Gridworld(
            cook_time=2,
            time_to_heat_oven=3,
            random_activation_prob=0.1,
            height=4,
            width=4,
        ),
    )
    gridworld_env.keyboard_control.run(env, actions=" swda", seed=0)
