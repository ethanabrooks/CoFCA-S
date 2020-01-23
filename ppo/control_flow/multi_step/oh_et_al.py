import ppo.control_flow.multi_step.abstract_recurrence as abstract_recurrence
import ppo.control_flow.oh_et_al as oh_et_al


class Recurrence(abstract_recurrence.Recurrence, oh_et_al.Recurrence):
    def __init__(
        self, hidden_size, gate_coef, conv_hidden_size, use_conv, nl_2, gate_h, **kwargs
    ):
        oh_et_al.Recurrence.__init__(
            self,
            hidden_size=hidden_size,
            gate_coef=gate_coef,
            use_conv=use_conv,
            conv_hidden_size=conv_hidden_size,
            nl_2=nl_2,
            gate_h=gate_h,
            **kwargs
        )
        abstract_recurrence.Recurrence.__init__(
            self,
            gate_coef=gate_coef,
            conv_hidden_size=conv_hidden_size,
            use_conv=use_conv,
            nl_2=nl_2,
            gate_h=gate_h,
        )
