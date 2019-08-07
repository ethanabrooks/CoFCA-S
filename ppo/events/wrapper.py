from collections import namedtuple, defaultdict
from typing import List

import numpy as np

import gym
from gym import spaces
from gym.wrappers import TimeLimit

from ppo.events.gridworld import State
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
    LetDogIn,
    KeepBabyOutOfFire,
    KeepCatFromDog,
)
from ppo.utils import RESET, REVERSE

Obs = namedtuple("Obs", "base subtasks")


class Wrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        watch_baby_range: float,
        avoid_dog_range: float,
        door_time_limit: int,
        max_time_outside: int,
        n_active_subtasks: int,
        subtasks: List[str] = None,
        check_obs=True,
    ):
        super().__init__(env)
        self.check_obs = check_obs
        self.n_active_subtasks = n_active_subtasks

        def make_subtasks():
            return filter(
                lambda s: (type(s).__name__ in subtasks) if subtasks else True,
                [
                    AnswerDoor(door_time_limit),
                    CatchMouse(),
                    ComfortBaby(),
                    MakeDinner(),  # TODO: failing
                    MakeFire(),
                    KillFlies(),
                    CleanMess(),
                    AvoidDog(avoid_dog_range),
                    WatchBaby(watch_baby_range),
                    LetDogIn(max_time_outside),
                    KeepBabyOutOfFire(),
                    KeepCatFromDog(),
                ],
            )

        self.make_subtasks = make_subtasks
        self.active_subtasks = None
        self.subtask_indexes = None
        self.rewards = None
        env = env.unwrapped
        self.random = env.random
        self.width, self.height = env.width, env.height
        self.pos_one_hots = np.eye(env.height * env.width)
        self.object_types = env.object_types
        self.obj_one_hots = np.eye(len(env.object_types))
        base_shape = len(self.object_types), self.height, self.width
        n_subtasks = len(list(make_subtasks()))
        subtasks_nvec = n_subtasks * np.ones(n_active_subtasks)
        assert n_active_subtasks <= n_subtasks
        self.observation_space = spaces.Dict(
            Obs(
                base=spaces.Box(
                    low=-2 * np.ones(base_shape), high=2 * np.ones(base_shape)
                ),
                subtasks=spaces.MultiDiscrete(subtasks_nvec),
            )._asdict()
        )
        self.action_space = spaces.Discrete(4 + len(env.object_types))

    def render(self, mode="human", pause=True, **kwargs):
        env = self.env.unwrapped
        legend = {}
        object_string = defaultdict(str)
        object_count = defaultdict(int)
        for obj in env.objects:
            if obj.pos is not None:
                string = "{:^3}".format(obj.icon())
                if obj.grasped:
                    string = REVERSE + string + RESET
                object_string[obj.pos] += string
                object_count[obj.pos] += len(string)
            legend[type(obj)] = obj
        # for v in legend.values():
        #     print("{:<15}".format(f"{v}:"), v.icon())

        width = max(max(object_count.values()), 10)
        object_string = {
            k: "{:^{width}}".format(v, width=width) for k, v in object_string.items()
        }

        for i in range(self.height):
            print("\n" + "-" * self.width * (1 + width))
            for j in range(self.width):
                print(object_string.get((i, j), " " * width), end="|")
        print()
        for subtask in self.active_subtasks:
            print(subtask, end="")
            if self.rewards is not None:
                print(": ", end="")
                try:
                    print(self.rewards[subtask])
                except KeyError:
                    print(0)
            else:
                print()
        if pause:
            input("pause")

    def reset(self, **kwargs):
        possible_subtasks = list(self.make_subtasks())
        self.subtask_indexes = np.random.choice(
            len(possible_subtasks), size=self.n_active_subtasks, replace=False
        )
        self.active_subtasks = [possible_subtasks[i] for i in self.subtask_indexes]
        return self.observation(super().reset())

    def get_rewards(self, s: State):
        object_dict = defaultdict(list)
        for obj in s.objects:
            k = type(obj).__name__.lower()
            object_dict[k] += [obj]
        object_dict = {k: v[0] if len(v) == 1 else v for k, v in object_dict.items()}
        subtask: Subtask
        for subtask in self.active_subtasks:
            subtask.step(*s.interactions, **object_dict)  # TODO wtf
            if subtask.condition(*s.interactions, **object_dict):
                yield subtask, subtask.reward

    def step(self, action):
        s, _, t, i = super().step(action)
        self.rewards = dict(self.get_rewards(s))
        return self.observation(s), sum(self.rewards.values()), t, i

    def observation(self, observation):
        dims = self.height, self.width
        object_pos = defaultdict(lambda: np.zeros((self.height, self.width)))
        for obj in observation.objects:
            if obj.pos is not None:
                index = np.ravel_multi_index(obj.pos, dims)
                one_hot = self.pos_one_hots[index].reshape(dims)
                c = -1 if obj.activated else 1
                if obj.grasped:
                    c *= 2

                object_pos[type(obj)] += c * one_hot
        base = np.stack([object_pos[k] for k in self.object_types])
        obs = Obs(base=base, subtasks=self.subtask_indexes)._asdict()
        if self.check_obs:
            assert self.observation_space.contains(obs)
        return obs


class BaseWrapper(Wrapper):
    def __init__(self, subtask, **kwargs):
        super().__init__(subtask, **kwargs)
        self.observation_space = self.observation_space.spaces["base"]

    def observation(self, observation):
        obs = super().observation(observation)["base"]
        if self.check_obs:
            assert self.observation_space.contains(obs)
        return obs


if __name__ == "__main__":
    import ppo.events.keyboard_control

    env = TimeLimit(
        max_episode_steps=30,
        env=Wrapper(
            n_active_subtasks=1,
            watch_baby_range=2,
            avoid_dog_range=2,
            door_time_limit=10,
            max_time_outside=15,
            env=ppo.events.Gridworld(
                cook_time=2,
                time_to_heat_oven=3,
                doorbell_prob=0.05,
                mouse_prob=0.2,
                baby_prob=0.1,
                mess_prob=0.02,
                fly_prob=0.005,
                height=4,
                width=4,
            ),
        ),
    )
    ppo.events.keyboard_control.run(env, actions="xswda1234567890", seed=0)
