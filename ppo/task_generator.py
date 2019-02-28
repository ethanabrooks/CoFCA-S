from ppo.util import NoInput
import torch


class TaskGenerator(NoInput):
    def __init__(self, task_size, learning_rate: float, entropy_coef: float,
                 **kwargs):
        super().__init__(task_size)
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.task_size = task_size
        self.softmax = torch.nn.Softmax(dim=-1)

    def probs(self):
        return self.softmax(self.weight).view(self.task_size)

    def importance_weight(self, task_index):
        return 1 / (self.task_size * self.probs()[task_index]).detach()
