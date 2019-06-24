import torch
import torch.jit
from torch.nn import functional as F

from ppo.distributions import FixedCategorical
from ppo.subtasks.agent import RecurrentState, sample_new
from ppo.subtasks.teacher import g123_to_binary
from ppo.utils import broadcast3d, interp
from ppo.subtasks.wrappers import Actions
import ppo


class Agent(ppo.subtasks.agent.Agent):
    def build_recurrent_module(self, **kwargs):
        return Recurrence(**kwargs)


class Recurrence(ppo.subtasks.agent.Recurrence):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs, hx):
        assert hx is not None
        T, N, D = inputs.shape
        # noinspection PyProtectedMember
        n_actions = len(Actions._fields)
        inputs, *actions = torch.split(
            inputs.detach(), [D - n_actions] + [1] * n_actions, dim=2)
        actions = Actions(*actions)
        obs, _, task, next_subtask = torch.split(
            inputs, self.obs_sections, dim=2)
        obs = obs.view(T, N, *self.obs_shape)
        task = task.view(T, N, self.n_subtasks,
                         self.obs_spaces.subtask.nvec.size)
        task = torch.split(task, 1, dim=-1)
        interaction, count, obj = [x[0, :, :, 0] for x in task]
        M123 = torch.stack([interaction, count, obj], dim=-1)
        one_hots = [self.part0_one_hot, self.part1_one_hot, self.part2_one_hot]
        g123 = (interaction, count, obj)
        M = g123_to_binary(g123, one_hots)
        new_episode = torch.all(hx.squeeze(0) == 0, dim=-1)
        hx = self.parse_hidden(hx)

        p = hx.p
        r = hx.r
        float_subtask = hx.subtask
        for x in hx:
            x.squeeze_(0)

        if torch.any(new_episode):
            p[new_episode, 0] = 1.  # initialize pointer to first subtask
            r[new_episode] = M[new_episode, 0]  # initialize r to first subtask

            # initialize g to first subtask
            hx.g[new_episode] = 0.

        A = torch.cat([hx.a.unsqueeze(0), actions.a], dim=0).long().squeeze(2)
        G = torch.cat([hx.g.unsqueeze(0), actions.g], dim=0).long().squeeze(2)

        outputs = RecurrentState(*[[] for _ in RecurrentState._fields])

        for t in range(T):
            subtask = float_subtask.long()
            float_subtask += next_subtask[t]
            outputs.subtask.append(float_subtask)

            agent_layer = obs[t, :, 6, :, :].long()
            j, k, l = torch.split(agent_layer.nonzero(), 1, dim=-1)

            def phi_update(subtask_param, values, losses, probs):
                debug_obs = obs[t, j, :, k, l].squeeze(1)
                task_sections = torch.split(
                    subtask_param, tuple(self.obs_spaces.subtask.nvec), dim=-1)
                parts = (debug_obs, self.a_one_hots[A[t]]) + task_sections
                if self.multiplicative_interaction:
                    return self.f(parts)
                obs4d = 1
                for i1, part in enumerate(parts):
                    for i2 in range(len(parts)):
                        if i1 != i2:
                            part.unsqueeze_(i2 + 1)
                    obs4d = obs4d * part

                c_logits = self.phi_update(obs4d.view(N, -1))
                if self.hard_update:
                    c_dist = FixedCategorical(logits=c_logits)
                    c = actions.c[t]
                    sample_new(c, c_dist)
                    values.append(c)
                    probs.append(c_dist.probs)
                    losses.append(-c_dist.log_probs(next_subtask[t]))
                else:
                    c = torch.sigmoid(c_logits[:, :1])
                    values.append(c)
                    probs.append(torch.zeros_like(c_logits))  # dummy value
                    losses.append(
                        F.binary_cross_entropy(
                            torch.clamp(c, 0., 1.),
                            next_subtask[t],
                            reduction='none'))
                return c

            # c
            cr = phi_update(
                subtask_param=r,
                values=outputs.cr,
                losses=outputs.cr_loss,
                probs=outputs.cr_probs)
            g_binary = M[torch.arange(N), G[t]]
            cg = phi_update(
                subtask_param=g_binary,
                values=outputs.cg,
                losses=outputs.cg_loss,
                probs=outputs.cg_probs)

            # p
            p2 = F.pad(p, [1, 0], 'constant', 0)[:, :-1]
            p2[:, -1] += 1 - p2.sum(dim=-1)
            p = interp(p, p2, cr)
            outputs.p.append(p)

            # r
            r = (p.unsqueeze(1) @ M).squeeze(1)
            outputs.r.append(r)

            # g
            old_g = self.g_one_hots[G[t]]
            dist = FixedCategorical(
                probs=torch.clamp(interp(old_g, p, cg), 0., 1.))
            sample_new(G[t + 1], dist)
            outputs.g.append(G[t + 1])
            outputs.g_probs.append(dist.probs)
            outputs.g_loss.append(-dist.log_probs(subtask))

            # a
            idxs = torch.arange(N), G[t + 1]
            g_binary = M[idxs]
            conv_out = self.conv((obs[t],
                                  broadcast3d(g_binary, self.obs_shape[1:])))
            if self.agent is None:
                dist = self.actor(conv_out)
            else:
                g123 = M123[idxs]
                agent_inputs = torch.cat([
                    obs[t].view(N, -1), g123,
                    self.agent_dummy_values.expand(N, -1)
                ],
                                         dim=1)
                dist = self.agent(agent_inputs, rnn_hxs=None, masks=None).dist
            sample_new(A[t + 1], dist)
            # a[:] = 'wsadeq'.index(input('act:'))

            outputs.a.append(A[t + 1])
            outputs.a_probs.append(dist.probs)

            # v
            outputs.v.append(self.critic(conv_out))

        # for name, x in zip(RecurrentState._fields, outputs):
        #     if not x:
        #         print(name)
        #         import ipdb
        #         ipdb.set_trace()

        stacked = [torch.stack(x) for x in outputs]
        stacked = [x.float().view(*x.shape[:2], -1) for x in stacked]

        # for name, x, size in zip(RecurrentState._fields, stacked,
        #                          self.state_sizes):
        #     if x.size(2) != size:
        #         print(name, x, size)
        #         import ipdb
        #         ipdb.set_trace()
        #     if x.dtype != torch.float32:
        #         print(name)
        #         import ipdb
        #         ipdb.set_trace()

        hx = torch.cat(stacked, dim=-1)
        return hx, hx[-1]
