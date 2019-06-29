#! /usr/bin/env python
# stdlib
from collections import namedtuple
import sys
from typing import Container, Dict, Iterable, List

# third party
from gym import utils
import numpy as np
from rl_utils import cartesian_product
from six import StringIO

from gridworld_env.abstract_gridworld import AbstractGridWorld
from gridworld_env.discrete import DiscreteEnv

Transition = namedtuple("Transition", "probability new_state reward terminal")


class GridWorld(AbstractGridWorld, DiscreteEnv):
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
        super().__init__(
            nS=text_map.size,
            nA=len(self.actions),
            P=self.get_transitions(desc=text_map),
            isd=self.get_isd(desc=text_map),
        )
        self.transition_array = np.stack([t[0] for t in self.transitions])

    @property
    def transition_strings(self):
        return "🛑👉👇👈👆"

    def assign(self, **assignments):
        new_desc = self.original_desc.copy()
        for letter, new_states in assignments.items():
            states_ = [self.decode(i) for i in new_states]
            idxs = tuple(zip(*states_))
            new_desc[idxs] = letter
        self.desc = new_desc
        self.set_desc(self.desc)

    def set_desc(self, desc):
        self.P = self.get_transitions(desc)
        self.isd = self.get_isd(desc)
        self.last_transition = None  # for rendering
        self.nS = desc.size

    def get_isd(self, desc):
        isd = np.isin(desc, tuple(self.start))
        if isd.sum():
            return np.reshape(isd / isd.sum(), -1)
        return np.arange(self.nS)

    def get_transitions(self, desc):
        shape = desc.shape

        def get_state_transitions():
            product = cartesian_product(*map(np.arange, shape))
            for idxs in product:
                state = self.encode(*idxs)
                yield state, dict(get_action_transitions_from(state))

        def get_action_transitions_from(state: int):
            for action in self.actions:
                yield action, list(get_transition_tuples_from(state, action))

        def get_transition_tuples_from(state, action):
            coord = self.decode(state)
            for transition, probability in zip(
                self.transitions[action], self.probabilities[action]
            ):

                new_coord = np.clip(
                    np.array(coord) + transition,
                    a_min=np.zeros_like(coord, dtype=int),
                    a_max=np.array(self.desc.shape, dtype=int) - 1,
                )
                new_char = self.desc[tuple(new_coord)]

                if np.all(np.isin(new_char, self.blocked)):
                    new_coord = coord
                yield Transition(
                    probability=probability,
                    new_state=self.encode(*new_coord),
                    reward=self.reward.get(new_char, 0),
                    terminal=new_char in self.terminal,
                )

        return dict(get_state_transitions())

    def step(self, a):
        prev = self.decode(self.s)
        s, r, t, i = super().step(a)
        self.last_action = a
        self.last_reward = r
        self.last_transition = np.array(self.decode(s)) - np.array(prev)
        return s, r, t, i

    def reset(self):
        self.last_transition = None
        self.last_action = None
        return super().reset()

    def render_map(self, mode):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        out = self.desc.copy().tolist()
        i, j = self.decode(self.s)

        out[i][j] = utils.colorize(out[i][j], "blue", highlight=True)

        print("#" * (len(out[0]) + 2))
        for row in out:
            print("#" + "".join(row) + "#")
        print("#" * (len(out[0]) + 2))
        # No need to return anything for human
        if mode != "human":
            return outfile
        out[i][j] = self.desc[i, j]  # TODO: delete?

    def render(self, mode="human"):
        if self.last_transition is not None:
            idx = np.all(self.transition_array == self.last_transition, axis=1)
            transition_string = self.transition_strings[idx]
            print("transition:", transition_string.item())
        if self.last_action is not None:
            print("action:", self.transition_strings[self.last_action].item())
        if self.last_reward is not None:
            print("Reward:", self.last_reward)

        self.render_map(mode)

    def encode(self, *idxs):
        return np.ravel_multi_index(idxs, self.desc.shape)

    def decode(self, s):
        return np.unravel_index(s, self.desc.shape)

    def generate_matrices(self):
        self._transition_matrix = np.zeros((self.nS, self.nA, self.nS))
        self._reward_matrix = np.zeros((self.nS, self.nA, self.nS))
        for s1, action_P in self.P.items():
            for a, transitions in action_P.items():
                trans: Transition
                for trans in transitions:
                    self._transition_matrix[s1, a, trans.new_state] = trans.probability
                    self._reward_matrix[s1, a] = trans.reward
                    if trans.terminal:
                        for a in range(self.nA):
                            self._transition_matrix[
                                trans.new_state, a, trans.new_state
                            ] = 1
                            self._reward_matrix[trans.new_state, a] = 0
                            assert not np.any(self._transition_matrix > 1)

    @property
    def transition_matrix(self) -> np.ndarray:
        if self._transition_matrix is None:
            self.generate_matrices()
        return self._transition_matrix

    @property
    def reward_matrix(self) -> np.ndarray:
        if self._reward_matrix is None:
            self.generate_matrices()
        return self._reward_matrix


if __name__ == "__main__":
    import gym
    from gridworld_env.random_walk import run

    run(gym.make("BookGridGridWorld-v0"))
