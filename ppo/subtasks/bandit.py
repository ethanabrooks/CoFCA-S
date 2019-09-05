from ppo.subtasks import control_flow, keyboard_control


class Bandit(control_flow.Env):
    def __init__(self, seed, n_lines, n_arms, flip_prob):
        super().__init__(seed, n_lines)
        self.flip_prob = flip_prob
        self.n_arms = n_arms
        self.condition_bit = None
        self.arms = None

    def reset(self):
        self.condition_bit = self.random.randint(0, 2)
        self.arms = self.random.randint(low=0, high=self.n_arms, size=self.n_lines)
        return super().reset()

    def step(self, action):
        if action != self.arms[self.active]:
            return None, -1, True, {}
        if self.random.rand() < self.flip_prob:
            self.condition_bit = 1 - self.condition_bit
        return super().step(action)

    def get_observation(self):
        return [self.condition_bit]

    def evaluate_condition(self):
        return bool(self.condition_bit)

    def done(self):
        return True  # bandits are always done

    def line_strings(self, index, indent):
        line = self.lines[index]
        print(' ' * indent + line)
        if



    def render(self, mode="human"):
        for line in self.lines:





if __name__ == "__main__":
    keyboard_control.run(
        Bandit(seed=0, n_lines=6, n_arms=3, flip_prob=0.1), actions="123"
    )
