from abc import ABC
from typing import Container, Dict, Iterable, List

import gym
import numpy as np


class AbstractGridWorld(gym.Env, ABC):
    def __init__(
        self,
        text_map: Iterable[Iterable[str]],
        terminal: Container[str],
        reward: Dict[str, float],
        transitions: List[np.ndarray] = None,
        probabilities: List[np.ndarray] = None,
        start: Iterable[str] = "",
        blocked: Container[str] = "",
    ):

        if transitions is None:
            transitions = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        # because every action technically corresponds to a _list_ of transitions (to
        # permit for stochasticity, we add an additional level to the nested list
        # if necessary
        transitions = [t if isinstance(t[0], list) else [t] for t in transitions]

        if probabilities is None:
            probabilities = [[1]] * len(transitions)
        self.actions = list(range(len(transitions)))
        assert len(transitions) == len(probabilities)
        for i in range(len(transitions)):
            assert len(transitions[i]) == len(probabilities[i])
            assert sum(probabilities[i]) == 1
        self.transitions = transitions
        self.probabilities = probabilities
        self.terminal = np.array(list(terminal))
        self.blocked = np.array(list(blocked))
        self.start = np.array(list(start))
        self.reward = reward

        self.last_reward = None
        self.last_transition = None
        self.last_action = None
        self._transition_matrix = None
        self._reward_matrix = None

        self.desc = text_map = np.array([list(r) for r in text_map])  # type: np.ndarray

        self.original_desc = self.desc.copy()
