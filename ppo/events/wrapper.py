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
                # CatchMouse(),
                # ComfortBaby(),
                # MakeDinner(),
                # MakeFire(),
                # KillFlies(),
                # CleanMess(),
                # AvoidDog(avoid_dog_range),
                # WatchBaby(watch_baby_range),
                # LetDogIn(max_time_outside),
                # KeepBabyOutOfFire(),
                # KeepCatFromDog(),
            ]

        self.make_subtasks = make_subtasks
        self.active_mask = None
        self.active_subtasks = None
        self.rewards = None
        env = env.unwrapped
        self.width, self.height = env.width, env.height
        self.object_one_hots = np.eye(env.height * env.width)
        self.object_types = {o.__class__ for o in env.make_objects()}
        base_shape = len(self.object_types), self.height, self.width
        self.observation_space = spaces.Box(
            low=-np.ones(base_shape), high=np.ones(base_shape)
        )
        # self.observation_space = spaces.Dict(
        #     Obs(
        #         base=self.observation_space,
        #         subtasks=spaces.MultiBinary(len(make_subtasks())),
        #     )._asdict()
        # )
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

    def reset(self, **kwargs):
        possible_subtasks = self.make_subtasks()
        active_subtasks = np.random.choice(
            len(possible_subtasks), size=self.n_active_subtasks, replace=False
        )
        self.active_mask = np.isin(np.arange(len(possible_subtasks)), active_subtasks)
        self.active_subtasks = [possible_subtasks[i] for i in active_subtasks]
        return self.observation(super().reset())

    def get_rewards(self, s: State):
        object_dict = defaultdict(list)
        for obj in s.objects:
            k = str(obj)
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
                sign = -1 if obj.activated else 1
                index = np.ravel_multi_index(obj.pos, dims)
                one_hot = self.object_one_hots[index].reshape(dims)
                object_pos[obj.__class__] += one_hot * sign
        base = np.stack([object_pos[k] for k in self.object_types])
        # obs = Obs(base=base, subtasks=self.active_mask)._asdict()
        obs = base
        assert self.observation_space.contains(obs)
        return obs


if __name__ == "__main__":
    import gridworld_env.keyboard_control

    env = TimeLimit(
        max_episode_steps=30,
        env=Wrapper(
            n_active_subtasks=5,
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
