from ppo.subtasks import control_flow, keyboard_control


class Bandit(control_flow.Env):
    def __init__(self, seed, n_subtasks, n_arms, flip_prob):
        super().__init__(seed, n_subtasks)
        self.flip_prob = flip_prob
        self.n_arms = n_arms
        self.condition_bit = None

    def reset(self):
        self.condition_bit = self.random.randint(0, 2)
        return super().reset()

    def step(self, action):
        if self.random.rand() < self.flip_prob:
            self.condition_bit = 1 - self.condition_bit
        s, r, t, i = super().step(action)

    def get_observation(self):
        return [self.condition_bit]

    def evaluate_condition(self):
        return bool(self.condition_bit)

    def done(self):
        return True  # bandits are always done


if __name__ == "__main__":
    keyboard_control.run(
        Bandit(seed=0, n_subtasks=6, n_arms=3, flip_prob=0.1), actions="123"
    )
