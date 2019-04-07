import mujoco_py
import glfw
from utils import space_to_size, argparse, parse_groups
import numpy as np

import hsr
from ppo.env_adapter import HSREnv
from ppo.main import add_hsr_args


class ControlViewer(mujoco_py.MjViewer):
    def __init__(self, sim):
        super().__init__(sim)
        self.active_joint = 0
        self.delta = None

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        if key == glfw.KEY_0:
            self.active_joint = 0
        elif key == glfw.KEY_1:
            self.active_joint = 1
        elif key == glfw.KEY_2:
            self.active_joint = 2
        elif key == glfw.KEY_3:
            self.active_joint = 3
        elif key == glfw.KEY_4:
            self.active_joint = 4
        elif key == glfw.KEY_5:
            self.active_joint = 5
        elif key == glfw.KEY_6:
            self.active_joint = 6
        elif key == glfw.KEY_7:
            self.active_joint = 7
        elif key == glfw.KEY_8:
            self.active_joint = 8
        elif key == glfw.KEY_9:
            self.active_joint = 9
        elif key == glfw.KEY_LEFT_CONTROL:
            x, y = glfw.get_cursor_pos(window)
            self.delta = self._last_mouse_x - x, self._last_mouse_y - y

    def reset_delta(self):
        self.delta = None


class ControlHSREnv(HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def viewer_setup(self):
        self.viewer = ControlViewer(self.sim)

    def control_agent(self):
        self.render()
        if self.viewer.delta is not None:
            action = np.zeros(space_to_size(self.action_space))
            action[self.viewer.active_joint] = 1
            self.step(action)
        self.viewer.reset_delta()


def main(max_episode_steps, env_args):
    env = ControlHSREnv(**env_args)
    done = True

    while True:
        if done:
            env.reset()
        env.control_agent()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_hsr_args(parser)
    hsr.util.env_wrapper(main)(**parse_groups(parser))
