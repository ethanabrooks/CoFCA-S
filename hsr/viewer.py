import mujoco_py
import glfw

from hsr import HSREnv


class ControlViewer(mujoco_py.MjViewer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_joint = None

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


class ControlHSREnv(HSREnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def viewer_setup(self):
        self.viewer = ControlViewer(self.sim)


def main(env_args):
    env = ControlHSREnv(**env_args)
    done = True

    while True:
        if done:
            env.reset()


