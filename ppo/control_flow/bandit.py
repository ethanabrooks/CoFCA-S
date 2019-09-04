from ppo import control_flow


class Bandit(control_flow.Env):
    def __init__(self, n_subtasks):
        super().__init__()
        self.n_subtasks = n_subtasks
        self.subtasks = None

    def reset(self):
        self.subtasks = []
        for _ in range(self.n_subtasks):
            pass

    def next_subtask(self):
        pass

    def initial_subtask(self):
        pass

    def get_observation(self):
        pass
