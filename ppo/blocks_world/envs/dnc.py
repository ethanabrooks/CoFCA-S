from ppo import blocks_world


class Env(blocks_world.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
