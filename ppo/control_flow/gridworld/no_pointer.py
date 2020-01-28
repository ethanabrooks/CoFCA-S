import ppo.control_flow.gridworld.abstract_recurrence as abstract_recurrence
import ppo.control_flow.no_pointer as no_pointer


class Recurrence(abstract_recurrence.Recurrence, no_pointer.Recurrence):
    def __init__(self, hidden_size, conv_hidden_size, use_conv, **kwargs):
        self.conv_hidden_size = conv_hidden_size
        no_pointer.Recurrence.__init__(self, hidden_size=hidden_size, **kwargs)
        abstract_recurrence.Recurrence.__init__(
            self, conv_hidden_size=conv_hidden_size, use_conv=use_conv
        )

    @property
    def gru_in_size(self):
        return self.conv_hidden_size + 2 * self.encoder_hidden_size + self.hidden_size
