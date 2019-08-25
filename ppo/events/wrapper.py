from collections import namedtuple, defaultdict
from typing import List

import re
import numpy as np

import gym
from gym import spaces
from gym.wrappers import TimeLimit

from ppo.events.gridworld import State
from ppo.events.instructions import (
    AnswerDoor,
    CatchMouse,
    ComfortBaby,
    MakeDinner,
    MakeFire,
    KillFlies,
    CleanMess,
    AvoidDog,
    WatchBaby,
    Instruction,
    LetDogIn,
    KeepBabyOutOfFire,
    KeepCatFromDog,
)
from ppo.utils import RESET, REVERSE
from ppo.events.objects import Agent, Mess
import itertools

Obs = namedtuple("Obs", "base instructions interactable")


class Wrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        watch_baby_range: float,
        avoid_dog_range: float,
        door_time_limit: int,
        max_time_outside: int,
        instructions_per_task: int,
        evaluation: bool,
        measure_interactivity: bool,
        seed: int,
        instructions: List[str] = None,
        check_obs=True,
        test: List[List[str]] = None,
        valid: List[List[str]] = None,
    ):
        super().__init__(env)
        self.measure_interactivity = measure_interactivity
        self.testing = evaluation
        self.agent_index = env.object_types.index(Agent)
        self.check_obs = check_obs
        self.instructions_per_task = instructions_per_task

        def instruction_str(instruction):
            return type(instruction).__name__

        def make_instructions():
            return filter(
                lambda s: (instruction_str(s) in instructions)
                if instructions
                else True,
                [
                    AnswerDoor(door_time_limit),
                    CatchMouse(),
                    ComfortBaby(),
                    MakeDinner(),
                    MakeFire(),
                    KillFlies(),
                    CleanMess(),
                    AvoidDog(avoid_dog_range),
                    WatchBaby(watch_baby_range),
                    # LetDogIn(max_time_outside),
                    # KeepBabyOutOfFire(),
                    KeepCatFromDog(),
                ],
            )

        self.make_instructions = make_instructions
        self.active_instructions = None
        self.instruction_indexes = None
        self.rewards = None
        env = env.unwrapped
        self.seed, = env.seed(seed)
        self.random = env.random
        self.width, self.height = env.width, env.height
        self.pos_one_hots = np.eye(env.height * env.width)
        self.object_types = env.object_types
        self.obj_one_hots = np.eye(len(env.object_types))
        base_shape = len(self.object_types), self.height, self.width
        instructions = list(map(instruction_str, make_instructions()))
        n_instructions = len(instructions)
        self.test_set = [{instructions.index(s) for s in task} for task in test or []]
        self.valid_set = [{instructions.index(s) for s in task} for task in valid or []]
        assert instructions_per_task <= n_instructions
        self.observation_space = spaces.Dict(
            Obs(
                base=spaces.Box(
                    low=-2 * np.ones(base_shape),
                    high=2 * np.ones(base_shape),
                    dtype=float,
                ),
                instructions=spaces.MultiBinary(n_instructions),
                interactable=spaces.MultiBinary(len(self.object_types)),
            )._asdict()
        )
        self.action_space = spaces.Discrete(5 + len(env.object_types))
        self.test_returns = None
        self.split = None
        self.test_iter = 0
        self.instruction_one_hots = np.eye(n_instructions)

    def evaluate(self):
        self.testing = True
        self.split = 0
        self.test_returns = defaultdict(float)
        self.test_iter += 1

    def train(self):
        self.testing = False

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
        for instruction in self.active_instructions:
            print(instruction, end="")
            if self.rewards is not None:
                print(": ", end="")
                try:
                    print(self.rewards[instruction])
                except KeyError:
                    print(0)
            else:
                print()
        if pause:
            input("pause")

    def reset(self, **kwargs):
        possible_instructions = list(self.make_instructions())
        if self.testing:
            env = self.env.unwrapped
            env.seed(self.seed + self.test_iter)
            if self.measure_interactivity:
                allowed_instructions = list(
                    itertools.combinations(
                        range(len(possible_instructions)), self.instructions_per_task
                    )
                )
                self.instruction_indexes = list(
                    allowed_instructions[self.random.choice(len(allowed_instructions))]
                )
            else:
                self.instruction_indexes = list(
                    self.test_set[self.random.choice(len(self.test_set))]
                )
        else:
            exclude = self.test_set + self.valid_set
            combinations = itertools.combinations(
                range(len(possible_instructions)), self.instructions_per_task
            )
            allowed_instructions = [c for c in combinations if set(c) not in exclude]

            if self.measure_interactivity:
                # individual instructions:
                allowed_instructions += list(
                    set((i,) for c in allowed_instructions for i in c)
                )

            choice = self.random.choice(len(allowed_instructions))
            self.instruction_indexes = list(allowed_instructions[choice])

        # for i, s in enumerate(possible_instructions):
        # print(i, s)
        # self.instruction_indexes = np.array([3, 5])
        self.active_instructions = [
            possible_instructions[i] for i in self.instruction_indexes
        ]
        return self.observation(super().reset())

    def get_rewards(self, s: State):
        object_dict = defaultdict(list)
        for obj in s.objects:
            k = type(obj).__name__.lower()
            object_dict[k] += [obj]
        object_dict = {k: v[0] if len(v) == 1 else v for k, v in object_dict.items()}
        instruction: Instruction
        for instruction in self.active_instructions:
            instruction.step(*s.interactions, **object_dict)
            if instruction.condition(*s.interactions, **object_dict):
                yield instruction, instruction.reward

    def step(self, action):
        obs, _, t, logs = super().step(action)
        self.rewards = dict(self.get_rewards(obs))
        r = sum(self.rewards.values())
        if self.testing and self.measure_interactivity:
            self.test_returns[self.split] += r

            def format_split(split):
                x = round(split * 100)
                return x, 100 - x

            if t:
                splits = [i / 10 for i in range(11)]
                _return = self.test_returns[self.split]
                if self.split <= 1:
                    logs.update({f"return_{format_split(self.split)}": _return})
                else:
                    logs.update({"optimal_return": _return})

                self.split += 0.1

        return self.observation(obs), r, t, logs

    def observation(self, observation):
        dims = self.height, self.width
        object_pos = defaultdict(lambda: np.zeros((self.height, self.width)))
        interactable = np.zeros(len(self.object_types))
        env = self.env.unwrapped
        for obj in observation.objects:
            if obj.pos is not None:
                index = np.ravel_multi_index(obj.pos, dims)
                one_hot = self.pos_one_hots[index].reshape(dims)
                c = -1 if obj.activated else 1
                if obj.grasped:
                    c *= 2

                t = type(obj)
                object_pos[t] += c * one_hot
                if obj.pos == env.agent.pos and obj is not env.agent:
                    interactable[env.object_types.index(t)] = 1
        base = np.stack([object_pos[k] for k in self.object_types])
        if self.measure_interactivity and self.testing and self.split <= 1:
            instructions = [
                self.random.choice(
                    self.instruction_indexes, p=[self.split, 1 - self.split]
                )
            ]
        else:
            instructions = list(self.instruction_indexes)
        instructions = self.instruction_one_hots[instructions].sum(0)
        obs = Obs(
            base=base, instructions=instructions, interactable=interactable
        )._asdict()
        if self.check_obs:
            assert self.observation_space.contains(obs)
        return obs


if __name__ == "__main__":
    import ppo.events.keyboard_control
    import sys

    try:
        seed = sys.argv[1]
    except IndexError:
        seed = 0

    env = TimeLimit(
        max_episode_steps=30,
        env=Wrapper(
            instructions_per_task=1,
            watch_baby_range=2,
            avoid_dog_range=4,
            door_time_limit=16,
            max_time_outside=15,
            measure_interactivity=False,
            instructions=["MakeDinner"],
            env=ppo.events.Gridworld(
                cook_time=1,
                time_to_heat_oven=2,
                toward_cat_prob=0.5,
                mouse_prob=0.2,
                baby_prob=0.1,
                time_limit=30,
                mess_prob=0.1,
                fly_prob=0.05,
                height=8,
                width=8,
                seed=seed,
                baby_speed=0.1,
                cat_speed=0.1,
                dog_speed=0.2,
                fly_speed=0,
                mouse_speed=0.1,
                toward_fire_prob=0.7,
                toward_hole_prob=0.5,
            ),
            evaluation=False,
        ),
    )
    ppo.events.keyboard_control.run(env, actions="xswda1234567890")
