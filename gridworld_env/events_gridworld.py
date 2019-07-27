import gym
import numpy as np


class EventsGridworld(gym.Env):
    def __init__(self):
        self.objects = None
        self.pos = None
        self.transitions = None

    def step(self, a):
        self.last_action = a
        n_transitions = len(self.transitions)
        pos = tuple(self.pos)
        touching = pos in self.objects

        if a < n_transitions:
            # move
            pos = self.pos + self.transitions[a]
            a_min = np.zeros(2)
            a_max = np.array(self.desc.shape) - 1
            self.pos = np.clip(pos, a_min, a_max).astype(int)

        elif self.grasping is None and touching:
            obj = self.objects[pos]
            if obj in self.graspable:
                self.grasping = obj
            else:
                raise NotImplemented
        elif self.grasping is not None and not touching:
            # put down
            self.objects[pos] = self.grasping
            self.grasping = None




    def reset(self):
        pass

    def render(self, mode='human'):
        pass