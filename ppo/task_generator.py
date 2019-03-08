import torch

from ppo.util import NoInput, init_normc_


class TaskGenerator(NoInput):
    def __init__(self, task_size, learning_rate: float, entropy_coef: float,
                 **kwargs):
        super().__init__(task_size)
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.task_size = task_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = 10
        self.logits = torch.Tensor(1, task_size)
        init_normc_(self.logits)
        self.logits = self.logits.view(-1)

    def probs(self):
        return self.softmax(self.temperature * self.logits).view(
            self.task_size)

    def importance_weight(self, task_index):
        return 1 / (self.task_size * self.probs()[task_index]).detach()
