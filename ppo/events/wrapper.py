import functools
from collections import namedtuple, defaultdict
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
import ppo.events

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
    ):
        super().__init__(env)
        self.n_active_subtasks = n_active_subtasks

        def make_subtasks():
            return [
                AnswerDoor(door_time_limit),
                CatchMouse(),
                ComfortBaby(),
                MakeDinner(),
                MakeFire(),
                KillFlies(),
                CleanMess(),
                AvoidDog(avoid_dog_range),
                WatchBaby(watch_baby_range),
                LetDogIn(max_time_outside),
                KeepBabyOutOfFire(),
                KeepCatFromDog(),
            ]

        self.make_subtasks = make_subtasks
        self.active_subtasks = None
        self.active_subtask_idxs = None
        self.rewards = None
        env = env.unwrapped
        self.random = env.random
        self.width, self.height = env.width, env.height
        self.pos_one_hots = np.eye(env.height * env.width)
        self.object_types = env.object_types
        self.obj_one_hots = np.eye(len(env.object_types))
        base_shape = len(self.object_types), self.height, self.width
        subtasks_nvec = len(make_subtasks()) * np.ones(n_active_subtasks)
        self.observation_space = spaces.Dict(
            Obs(
                base=spaces.Box(low=np.zeros(base_shape), high=4 * np.ones(base_shape)),
                subtasks=spaces.MultiDiscrete(subtasks_nvec),
            )._asdict()
        )
        self.action_space = spaces.Discrete(5)

    def render(self, mode="human", **kwargs):
        env = self.env.unwrapped
        legend = {}
        object_string = defaultdict(str)
        object_count = defaultdict(int)
        for obj in env.objects:
            if obj.pos is not None:
                string = "{:^3}".format(obj.icon())
                object_string[obj.pos] += string
                object_count[obj.pos] += len(string)
            legend[obj.__class__] = obj
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
        print("Grasping", env.agent.grasping)
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
        input("pause")

    def reset(self, **kwargs):
        possible_subtasks = self.make_subtasks()
        self.active_subtask_idxs = np.random.choice(
            len(possible_subtasks), size=self.n_active_subtasks, replace=False
        )
        self.active_subtasks = [possible_subtasks[i] for i in self.active_subtask_idxs]
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
        grasping = self.env.unwrapped.agent.grasping
        pos = defaultdict(set)
        for obj in observation.objects:
            if obj.pos is not None:
                pos[type(obj)].add(obj.pos)
                index = np.ravel_multi_index(obj.pos, dims)
                one_hot = self.pos_one_hots[index].reshape(dims)
                if not obj.activated and not grasping is obj:
                    c = 1
                elif obj.activated and not grasping is obj:
                    c = 2
                elif not obj.activated and grasping is obj:
                    c = 3
                elif obj.activated and grasping is obj:
                    c = 4
                else:
                    raise RuntimeWarning

                x = c * one_hot
                if x.any() > 4:
                    import ipdb

                    ipdb.set_trace()
                object_pos[type(obj)] = x
        base = np.stack([object_pos[k] for k in self.object_types])
        obs = Obs(base=base, subtasks=self.active_subtask_idxs)._asdict()
        assert self.observation_space.contains(obs)
        return obs


if __name__ == "__main__":
    import gridworld_env.keyboard_control

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
    gridworld_env.keyboard_control.run(env, actions=" swda", seed=0)
