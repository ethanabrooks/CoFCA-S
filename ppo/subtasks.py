from collections import namedtuple

from gym.spaces import Box, Discrete
import numpy as np
import torch
import torch.jit
from gym import spaces
from gym.spaces import Box, Discrete
from torch import nn as nn
from torch.nn import functional as F

from ppo.agent import Agent, AgentValues, NNBase
from ppo.distributions import Categorical, DiagGaussian, FixedCategorical
from ppo.layers import Broadcast3d, Concat, Flatten, Parallel, Product, Reshape
from ppo.teacher import SubtasksTeacher
from ppo.utils import batch_conv1d, broadcast_3d, init_, interp, trace
from ppo.wrappers import SubtasksActions, get_subtasks_action_sections, get_subtasks_obs_sections

RecurrentState = namedtuple(
    'RecurrentState',
    'p r h b b_probs g_binary g_int g_probs c c_probs l l_probs a a_probs v c_truth '
    'c_loss l_loss p_loss r_loss g_loss b_loss subtask')


# noinspection PyMissingConstructor
class SubtasksAgent(Agent, NNBase):
    def __init__(self,
                 obs_shape,
                 action_space,
                 task_space,
                 hidden_size,
                 entropy_coef,
                 alpha,
                 zeta,
                 hard_update,
                 teacher_agent=None,
                 **kwargs):
        nn.Module.__init__(self)
        self.zeta = zeta
        self.alpha = alpha
        self.hard_update = hard_update
        if teacher_agent:
            assert isinstance(teacher_agent, SubtasksTeacher)
        self.teacher_agent = teacher_agent
        self.entropy_coef = entropy_coef
        self.action_space = SubtasksActions(*action_space.spaces)
        self.recurrent_module = SubtasksRecurrence(
            obs_shape=obs_shape,
            action_space=self.action_space,
            task_space=task_space,
            hidden_size=hidden_size,
            hard_update=hard_update,
            **kwargs,
        )
        self.obs_sections = get_subtasks_obs_sections(task_space)
        self.register_buffer(
            'subtask_choices',
            torch.zeros(
                self.action_space.g_int.n,
                self.action_space.g_int.n,
                dtype=torch.long))

    def forward(self, inputs, rnn_hxs, masks, action=None,
                deterministic=False):
        obs, subtask, task, next_subtask = torch.split(
            inputs, self.obs_sections, dim=1)

        n = inputs.size(0)
        actions = None
        if action is not None:
            action_sections = get_subtasks_action_sections(self.action_space)
            actions = SubtasksActions(
                *torch.split(action, action_sections, dim=-1))

        all_hxs, last_hx = self._forward_gru(
            inputs.view(n, -1), rnn_hxs, masks, actions=actions)
        rm = self.recurrent_module
        hx = RecurrentState(*rm.parse_hidden(all_hxs))

        # print('g       ', hx.g[0])
        # print('g_target', g_target[0, :, 0, 0])

        if self.hard_update:
            dists = SubtasksActions(
                a=FixedCategorical(hx.a_probs),
                b=FixedCategorical(hx.b_probs),
                c=FixedCategorical(hx.c_probs),
                l=FixedCategorical(hx.l_probs),
                g_int=FixedCategorical(hx.g_probs),
            )
        else:
            dists = SubtasksActions(
                a=FixedCategorical(hx.a_probs),
                b=FixedCategorical(hx.b_probs),
                c=None,
                l=None,
                g_int=FixedCategorical(hx.g_probs))

        if action is None:
            actions = SubtasksActions(
                a=hx.a, b=hx.b, l=hx.l, c=hx.c, g_int=hx.g_int)

        log_probs = sum(
            dist.log_probs(a) for dist, a in zip(dists, actions)
            if dist is not None)
        entropies = sum(dist.entropy() for dist in dists if dist is not None)

        if action is not None:
            subtask_int = rm.encode(subtask[:, :, 0, 0])
            codes = torch.unique(subtask_int)
            g_one_hots = rm.g_one_hots[actions.g_int.long().flatten()].long()
            for code in codes:
                idx = subtask_int == code
                self.subtask_choices[code] += g_one_hots[idx].sum(dim=0)

        c = hx.c.round()

        # For derivation, see https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
        choices = self.subtask_choices.float()
        cramers_v = torch.tensor(0.)
        n = choices.sum()
        if n > 0:
            ni = choices.sum(dim=0, keepdim=True)
            nj = choices.sum(dim=1, keepdim=True)
            Ei = ni * nj / n
            if torch.all(Ei > 0):
                chi_squared = torch.sum((choices - Ei)**2 / Ei)
                cramers_v = torch.sqrt(
                    chi_squared / n / self.action_space.g_int.n)

        log = dict(
            # g_accuracy=g_accuracy.float(),
            c_accuracy=(torch.mean((c == hx.c_truth).float())),
            c_recall=(torch.mean(
                (c[hx.c_truth > 0] == hx.c_truth[hx.c_truth > 0]).float())),
            c_precision=(torch.mean((c[c > 0] == hx.c_truth[c > 0]).float())),
            subtask_association=cramers_v)
        aux_loss = -self.entropy_coef * entropies.mean()

        if self.teacher_agent:
            imitation_dist = self.teacher_agent(inputs, rnn_hxs, masks).dist
            imitation_probs = imitation_dist.probs.detach().unsqueeze(1)
            our_log_probs = torch.log(dists.a.probs).unsqueeze(2)
            imitation_obj = (imitation_probs @ our_log_probs).view(-1)
            log.update(imitation_obj=imitation_obj)
            aux_loss -= torch.mean(imitation_obj)

        for k, v in hx._asdict().items():
            if k.endswith('_loss'):
                log[k] = v

        return AgentValues(
            value=hx.v,
            action=torch.cat(actions, dim=-1),
            action_log_probs=log_probs,
            aux_loss=aux_loss,
            rnn_hxs=torch.cat(hx, dim=-1),
            dist=None,
            log=log)

    def get_value(self, inputs, rnn_hxs, masks):
        n = inputs.size(0)
        all_hxs, last_hx = self._forward_gru(
            inputs.view(n, -1), rnn_hxs, masks)
        return self.recurrent_module.parse_hidden(all_hxs).v

    def _forward_gru(self, x, hxs, masks, actions=None):
        if actions is None:
            y = F.pad(x, (0, 2), 'constant', -1)
        else:
            y = torch.cat([x, actions.a, actions.g_int], dim=-1)
        return super()._forward_gru(y, hxs, masks)

    @property
    def recurrent_hidden_state_size(self):
        return sum(self.recurrent_module.state_sizes)

    @property
    def is_recurrent(self):
        return True


class SubtasksRecurrence(torch.jit.ScriptModule):
    __constants__ = [
        'input_sections', 'subtask_space', 'state_sizes', 'recurrent'
    ]

    def __init__(self, obs_shape, action_space, task_space, hidden_size,
                 recurrent, hard_update, multiplicative_interaction):
        super().__init__()
        d, h, w = obs_shape
        conv_out_size = h * w * hidden_size
        subtask_space = list(map(int, task_space.nvec[0]))
        subtask_size = sum(subtask_space)
        n_subtasks = task_space.shape[0]
        self.hard_update = hard_update
        self.obs_sections = get_subtasks_obs_sections(task_space)
        self.obs_shape = d, h, w
        self.task_nvec = task_space.nvec
        self.action_space = action_space
        self.n_subtasks = n_subtasks

        # networks
        self.recurrent = recurrent

        self.conv1 = nn.Sequential(
            init_(
                nn.Conv2d(
                    self.obs_sections.base,
                    hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1), 'relu'), nn.ReLU(), Flatten())

        if multiplicative_interaction:
            conv_weight_shape = hidden_size, self.obs_sections.base, 3, 3
            self.conv_weight = nn.Sequential(
                nn.Linear(self.obs_sections.subtask,
                          np.prod(conv_weight_shape)),
                Reshape(-1, *conv_weight_shape))

        else:
            self.conv2 = nn.Sequential(
                Concat(dim=1),
                init_(
                    nn.Conv2d(
                        self.obs_sections.base + self.obs_sections.subtask,
                        hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding=1), 'relu'), nn.ReLU(), Flatten())

        self.f = nn.Sequential(
            Parallel(
                init_(nn.Linear(self.obs_sections.base, hidden_size)),
                init_(nn.Linear(action_space.a.n, hidden_size)),
                *[init_(nn.Linear(i, hidden_size)) for i in self.task_nvec[0]],
            ),
            Product(),
        )

        subcontroller = nn.GRUCell if recurrent else nn.Linear
        self.subcontroller = trace(
            lambda in_size: nn.Sequential(
                init_(subcontroller(in_size, hidden_size), 'relu'),
                nn.ReLU(),
            ),
            in_size=conv_out_size)  # h

        self.phi_update = trace(
            # lambda in_size: init_(nn.Linear(in_size, 2), 'sigmoid'),
            # in_size=(
            # hidden_size +  s
            # hidden_size))  h
            lambda in_size: init_(nn.Linear(in_size, 1), 'sigmoid'),
            in_size=hidden_size)

        self.phi_shift = trace(
            lambda in_size: nn.Sequential(
                # init_(nn.Linear(in_size, hidden_size), 'relu'),
                # nn.ReLU(),
                # init_(nn.Linear(hidden_size, 3)),  # 3 for {-1, 0, +1}
                init_(nn.Linear(in_size, 3)),  # 3 for {-1, 0, +1}
            ),
            # in_size=hidden_size)
            in_size=hidden_size)

        self.pi_theta = Categorical(
            subtask_size,  # r
            action_space.g_int.n)

        self.beta = Categorical(hidden_size, 2)
        input_size = h * w * hidden_size  # conv output
        if isinstance(action_space.a, Discrete):
            num_outputs = action_space.a.n
            self.actor = Categorical(input_size, num_outputs)
        elif isinstance(action_space.a, Box):
            num_outputs = action_space.a.size(0)
            self.actor = DiagGaussian(input_size, num_outputs)
        else:
            raise NotImplementedError
        self.critic = init_(nn.Linear(input_size, 1))

        # embeddings
        for name, d in zip(
            ['type_embeddings', 'count_embeddings', 'obj_embeddings'],
                subtask_space):
            self.register_buffer(name, torch.eye(int(d)))

        self.register_buffer('l_one_hots', torch.eye(3))
        self.register_buffer('p_one_hots', torch.eye(self.n_subtasks))
        self.register_buffer('a_one_hots', torch.eye(int(action_space.a.n)))
        self.register_buffer('g_one_hots', torch.eye(
            int(action_space.g_int.n))),
        self.register_buffer('subtask_space',
                             torch.tensor(task_space.nvec[0].astype(np.int64)))

        state_sizes = RecurrentState(
            p=self.n_subtasks,
            r=subtask_size,
            h=hidden_size,
            g_binary=subtask_size,
            g_int=1,
            b=1,
            b_probs=2,
            g_probs=action_space.g_int.n,
            c=1,
            c_truth=1,
            c_probs=2,
            l=1,
            a=1,
            v=1,
            a_probs=action_space.a.n,
            l_probs=3,
            c_loss=1,
            l_loss=1,
            p_loss=1,
            r_loss=1,
            g_loss=1,
            b_loss=1,
            subtask=1)
        self.state_sizes = RecurrentState(*map(int, state_sizes))

    # @torch.jit.script_method
    def parse_hidden(self, hx):
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    @torch.jit.script_method
    def task_one_hots(self, task_type, count, obj):
        return torch.cat([
            self.type_embeddings[task_type.long()],
            self.count_embeddings[count.long()],
            self.obj_embeddings[obj.long()],
        ],
                         dim=-1)

    def encode(self, g_binary):
        factored_code = g_binary.nonzero()[:, 1:].view(-1, 3)
        factored_code -= F.pad(
            torch.cumsum(self.subtask_space, dim=0)[:2], (1, 0), 'constant', 0)
        # numpy_codes = factord_code.clone().numpy()
        factored_code[:, :-1] *= self.subtask_space[1:]  # g1 * x2, g2 * x3
        factored_code[:, 0] *= self.subtask_space[2]  # g1 * x3
        codes = factored_code.sum(dim=-1)
        # codes1 = codes.numpy()
        # codes2 = np.ravel_multi_index(numpy_codes.T, (self.subtask_space.numpy()))
        # if not np.array_equal(codes1, codes2):
        #     import ipdb; ipdb.set_trace()
        return codes

    def decode(self, g):
        x1, x2, x3 = self.subtask_space.to(g.dtype)
        g1 = g // (x2 * x3)
        x4 = g % (x2 * x3)
        g2 = x4 // x3
        g3 = x4 % x3
        return g1, g2, g3

    def task_to_one_hot(self, g):
        return self.task_one_hots(*self.decode(g)).squeeze(1)

    def check_grad(self, **kwargs):
        for k, v in kwargs.items():
            if v.grad_fn is None:
                continue
            grads = torch.autograd.grad(
                v.mean(),
                self.parameters(),
                retain_graph=True,
                allow_unused=True)
            for (name, _), grad in zip(self.named_parameters(), grads):
                if grad is None:
                    print(f'{k} has no grad wrt {name}')
                else:
                    print(f'mean grad of {k} ({v.mean().item()}) wrt {name}:',
                          grad.mean())
                    if torch.isnan(grad.mean()):
                        import ipdb
                        ipdb.set_trace()

    # @torch.jit.script_method
    def forward(self, inputs, hx):
        assert hx is not None
        T, N, d = inputs.shape
        inputs = inputs.view(T, N, -1)
        inputs, a, g_int = torch.split(inputs, [d - 2, 1, 1], dim=2)
        inputs = inputs.view(T, N, *self.obs_shape)
        obs, subtasks, task, next_subtask = torch.split(
            inputs, self.obs_sections, dim=2)
        task = task[:, :, :, 0, 0]
        next_subtask = next_subtask[:, :, :, 0, 0]
        sections = [self.n_subtasks] * self.task_nvec.shape[1]
        task_type, count, obj = torch.split(task, sections, dim=-1)

        M = self.task_one_hots(task_type[0], (count - 1)[0], obj[0])
        new_episode = torch.all(hx.squeeze(0) == 0, dim=-1)
        hx = self.parse_hidden(hx)

        p = hx.p
        r = hx.r
        g_binary = hx.g_binary
        float_subtask = hx.subtask

        for x in hx:
            x.squeeze_(0)

        if torch.any(new_episode):
            p[new_episode, 0] = 1.  # initialize pointer to first subtask
            r[new_episode] = M[new_episode, 0]  # initialize r to first subtask

        outputs = RecurrentState(*[[] for _ in RecurrentState._fields])

        past_a = torch.cat([hx.a.unsqueeze(0), a], dim=0)

        n = obs.size(0)
        for i in range(n):
            float_subtask += next_subtask[i]
            outputs.subtask.append(float_subtask)
            subtask = float_subtask.long()
            m = M.size(0)

            # s = self.f(torch.cat([conv_out, r, g_binary, b], dim=-1))
            # logits = self.phi_update(torch.cat([s, h], dim=-1))
            # if self.hard_update:
            # dist = FixedCategorical(logits=logits)
            # c = dist.sample().float()
            # outputs.c_probs.append(dist.probs)
            # else:
            # c = torch.sigmoid(logits[:, :1])
            # outputs.c_probs.append(torch.zeros_like(logits))  # dummy value

            a_idxs = past_a[i].flatten().long()
            agent_layer = obs[i, :, 6, :, :].long()
            j, k, l = torch.split(agent_layer.nonzero(), [1, 1, 1], dim=-1)
            debug_obs = obs[i, j, :, k, l].squeeze(1)

            h = self.f((
                debug_obs,
                self.a_one_hots[a_idxs],
                *torch.split(g_binary, tuple(self.task_nvec[0]), dim=-1),
            ))

            c = torch.sigmoid(self.phi_update(h))
            outputs.c_truth.append(next_subtask[i])

            if torch.any(next_subtask[i] > 0):
                weight = torch.ones_like(c)
                weight[next_subtask[i] > 0] /= torch.sum(next_subtask[i] > 0)
                weight[next_subtask[i] == 0] /= torch.sum(next_subtask[i] == 0)

                outputs.c_loss.append(
                    F.binary_cross_entropy(
                        torch.clamp(c, 0., 1.),
                        next_subtask[i],
                        weight=weight,
                        reduction='none'))
            else:
                outputs.c_loss.append(torch.zeros_like(c))

            outputs.c.append(c)

            # TODO: figure this out
            # if self.recurrent:
            #     h2 = self.subcontroller(obs[i], h)
            # else:
            # h2 = self.subcontroller(conv_out)

            logits = self.phi_shift(h)
            # if self.hard_update:
            # dist = FixedCategorical(logits=logits)
            # l = dist.sample()
            # outputs.l.append(l.float())
            # outputs.l_probs.append(dist.probs)
            # l = self.l_one_hots[l]
            # else:
            l = F.softmax(logits, dim=1)
            outputs.l.append(torch.zeros_like(c))  # dummy value
            outputs.l_probs.append(torch.zeros_like(l))  # dummy value

            # l_loss
            l_target = 1 - next_subtask[i].long().flatten()
            outputs.l_loss.append(
                F.cross_entropy(
                    logits,
                    l_target,
                    reduction='none',
                ).unsqueeze(1))

            p2 = batch_conv1d(p, l)

            # p_losss
            outputs.p_loss.append(
                F.cross_entropy(
                    p2.squeeze(1), subtask.squeeze(1),
                    reduction='none').unsqueeze(1))

            r2 = p2 @ M

            # r_loss
            r_target = []
            for j in range(m):
                r_target.append(M[j, subtask[j]])
            r_target = torch.cat(r_target).detach()
            r_loss = F.binary_cross_entropy(
                torch.clamp(r2.squeeze(1), 0., 1.),
                r_target,
                reduction='none',
            )
            outputs.r_loss.append(torch.mean(r_loss, dim=-1, keepdim=True))

            p = interp(p, p2.squeeze(1), c)
            r = interp(r, r2.squeeze(1), c)

            # h = interp(h, h2, c)

            outputs.p.append(p)
            outputs.r.append(r)
            outputs.h.append(h)

            # TODO: deterministic
            # g
            dist = self.pi_theta(r)
            g_target = self.encode(M[torch.arange(m), subtask.flatten()])
            outputs.g_loss.append(-dist.log_probs(g_target))
            new = g_int[i] < 0
            g_int[i][new] = dist.sample()[new].float()
            outputs.g_int.append(g_int[i])
            outputs.g_probs.append(dist.probs)

            # g_loss
            # assert (int(i1), int(i2), int(i3)) == \
            #        np.unravel_index(int(g_int), self.subtask_space)
            g_binary2 = self.task_to_one_hot(g_int[i])
            g_binary = interp(g_binary, g_binary2, c)
            outputs.g_binary.append(g_binary)

            conv_out = self.conv1(obs[i])

            # b
            dist = self.beta(h)
            b = dist.sample().float()
            outputs.b_probs.append(dist.probs)
            outputs.c_probs.append(torch.zeros_like(dist.probs))  # TODO

            # b_loss
            outputs.b_loss.append(-dist.log_probs(next_subtask[i]))
            outputs.b.append(b)

            # a
            g_broad = broadcast_3d(g_binary, self.obs_shape[1:])
            conv_out2 = self.conv2((obs[i], g_broad))
            dist = self.actor(conv_out2)
            new = a[i] < 0
            a[i, new] = dist.sample()[new].float()
            # a[:] = 'wsadeq'.index(input('act:'))

            outputs.a.append(a[i])
            outputs.a_probs.append(dist.probs)

            # v
            outputs.v.append(self.critic(conv_out2))

        stacked = []
        for x in outputs:
            stacked.append(torch.stack(x))

        hx = torch.cat(stacked, dim=-1)
        return hx, hx[-1]
