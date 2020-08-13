from __future__ import print_function

import argparse
import csv
import itertools
import json
import random
import subprocess
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
from rl_utils import hierarchical_parse_args
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import ppo.control_flow.multi_step.env
from ppo.agent import Agent
from ppo.control_flow.env import Action
from ppo.control_flow.multi_step.env import Env
from ppo.control_flow.multi_step.env import Obs
from layers import Flatten
from utils import init_
from typing import Optional

MAX_LAYERS = 3

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def format_image(data, output, target):
    return torch.stack([data.sum(1)[0], target[0], output[0]], dim=0).unsqueeze(1)


X = namedtuple("X", "obs line lower")


class GridworldDataset(IterableDataset):
    def __init__(self, lower_level_config, lower_level_load_path, render, **kwargs):
        self.render = render
        self.env = Env(rank=0, lower_level="pretrained", **kwargs)
        with lower_level_config.open() as f:
            lower_level_params = json.load(f)
        observation_space = Obs(**self.env.observation_space.spaces)
        ll_action_space = spaces.Discrete(Action(*self.env.action_space.nvec).lower)
        self.lower_level = Agent(
            obs_spaces=observation_space,
            entropy_coef=0,
            action_space=ll_action_space,
            lower_level=True,
            num_layers=1,
            **lower_level_params,
        )
        state_dict = torch.load(lower_level_load_path, map_location="cpu")
        self.lower_level.load_state_dict(state_dict["agent"])
        print(f"Loaded lower_level from {lower_level_load_path}.")

    def __iter__(self):
        s = self.env.reset()
        while True:
            s = {
                k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                for k, v in s.items()
            }
            S = Obs(**s)
            agent_values = self.lower_level(S, rnn_hxs=None, masks=None)
            lower = agent_values.action
            s, _, t, i = self.env.step(lower.cpu().numpy())
            if self.render:
                self.env.render()
            complete = i["subtask_complete"]
            if t:
                s = self.env.reset()
            else:
                active = S.active.long().item()
                yield X(
                    obs=S.obs.squeeze(0),
                    line=S.lines.squeeze(0)[active],
                    lower=lower.squeeze(0),
                ), complete

    # def __len__(self):
    #     pass


class Network(nn.Module):
    def __init__(
        self,
        d,
        h,
        w,
        line_nvec,
        conv_layers,
        conv_kernels,
        conv_strides,
        pool_type,
        pool_kernels,
        pool_strides,
        action_size,
        line_hidden_size,
        lower_hidden_size,
        concat,
    ):
        super().__init__()
        self.concat = concat
        if not concat:
            hidden_size = lower_hidden_size

        def remove_none(xs):
            return [x for x in xs if x is not None]

        conv_layers = remove_none(conv_layers)
        conv_kernels = remove_none(conv_kernels)
        conv_strides = remove_none(conv_strides)
        pool_kernels = remove_none(pool_kernels)
        pool_strides = remove_none(pool_strides)

        def generate_pools(k):
            for (kernel, stride) in zip(pool_kernels, pool_strides):
                kernel = min(k, kernel)
                padding = (kernel // 2) % stride
                if pool_type == "avg":
                    pool = nn.AvgPool2d(
                        kernel_size=kernel, stride=stride, padding=padding
                    )
                elif pool_type == "max":
                    pool = nn.MaxPool2d(
                        kernel_size=kernel, stride=stride, padding=padding
                    )
                else:
                    raise RuntimeError
                k = int((k + 2 * padding - kernel) / stride + 1)
                k = yield k, pool

        def generate_convolutions(k):
            in_size = d
            for (layer, kernel, stride) in zip(conv_layers, conv_kernels, conv_strides):
                kernel = min(k, kernel)
                padding = (kernel // 2) % stride
                conv = init_(
                    nn.Conv2d(
                        in_channels=in_size,
                        out_channels=layer,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                    )
                )
                k = int((k + (2 * padding) - (kernel - 1) - 1) // stride + 1)
                k = yield k, conv
                in_size = layer

        def generate_modules(k):
            n_pools = min(len(pool_strides), len(pool_kernels))
            n_conv = min(len(conv_layers), len(conv_strides), len(conv_kernels))
            conv_iterator = generate_convolutions(k)
            try:
                k, conv = next(conv_iterator)
                yield conv
                pool_iterator = None
                for i in itertools.count():
                    if pool_iterator is None:
                        if i >= n_conv - n_pools and pool_type is not None:
                            pool_iterator = generate_pools(k)
                            k, pool = next(pool_iterator)
                            yield pool
                    else:
                        k, pool = pool_iterator.send(k)
                        yield pool
                    k, conv = conv_iterator.send(k)
                    yield conv
            except StopIteration:
                pass
            out_size = k ** 2 * conv.out_channels
            yield Flatten(out_size=out_size)
            if not concat:
                yield init_(nn.Linear(out_size, hidden_size))

        *obs_modules, flatten = generate_modules(h)
        self.conv = nn.Sequential(*obs_modules, flatten)
        if not concat:
            line_hidden_size = hidden_size
            lower_hidden_size = hidden_size
        offset = F.pad(line_nvec.cumsum(0), [1, 0])
        self.register_buffer("offset", offset)
        self.embed_line = nn.EmbeddingBag(line_nvec.sum(), line_hidden_size)
        self.embed_lower = nn.Embedding(action_size, lower_hidden_size)
        self.out = init_(
            nn.Linear(
                flatten.out_size + line_hidden_size + lower_hidden_size
                if concat
                else hidden_size,
                1,
            )
        )

    def forward(self, x: X) -> torch.Tensor:
        embedded_line = self.embed_line(x.line.long())
        embedded_lower = self.embed_lower(x.lower.long().flatten())
        embedded_obs = self.conv(x.obs)
        if self.concat:
            h = torch.cat([embedded_line, embedded_lower, embedded_obs], dim=-1)
        else:
            h = embedded_lower * embedded_obs * embedded_line
        return self.out(h).sigmoid()


def main(
    no_cuda: bool,
    seed: int,
    batch_size: int,
    lr: float,
    log_dir: Path,
    run_id: str,
    dataset_args: dict,
    network_args: dict,
    log_interval: int,
    save_interval: int,
    load_path: Optional[Path],
):
    use_cuda = not (no_cuda or load_path) and torch.cuda.is_available()
    writer = SummaryWriter(str(log_dir))

    torch.manual_seed(seed)

    if use_cuda:
        nvidia_smi = subprocess.check_output(
            "nvidia-smi --format=csv --query-gpu=memory.free".split(),
            universal_newlines=True,
        )
        n_gpu = len(list(csv.reader(StringIO(nvidia_smi)))) - 1
        try:
            index = int(run_id[-1])
        except (ValueError, IndexError):
            index = random.randrange(0, n_gpu)
        print("Using GPU", index)
        device = torch.device("cuda", index=index % n_gpu)
    else:
        device = "cpu"

    dataset = GridworldDataset(**dataset_args, render=load_path is not None)
    env = dataset.env
    obs_spaces = Obs(**env.observation_space.spaces)
    obs_shape = obs_spaces.obs.shape
    action_spaces = Action(*env.action_space.nvec)
    network = Network(
        *obs_shape,
        line_nvec=torch.tensor(obs_spaces.lines.nvec[0]),
        action_size=action_spaces.lower,
        **network_args,
    )
    if load_path:
        state_dict = torch.load(load_path, map_location="cpu")
        network.load_state_dict(state_dict)
        batch_size = 1

    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    network.train()
    start = 0

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        **(dict(num_workers=1, pin_memory=True) if use_cuda else dict()),
    )
    total_loss = 0
    log_progress = None
    for i, (data, target) in enumerate(train_loader):
        target = target.to(device).float()
        data = X(*(x.to(device).float() for x in data))
        optimizer.zero_grad()
        out0 = network.conv[0](data.obs)
        out1 = network.conv[1](out0)
        out2 = network.conv[2](out1)
        out3 = network.conv[3](out2)
        print(network.conv)
        print(out0.shape)
        print(out1.shape)
        print(out2.shape)
        print(out3.shape)
        import ipdb

        ipdb.set_trace()
        output = network(data).flatten()
        if load_path is not None:
            print("output", output)
        loss = F.binary_cross_entropy(output, target, reduction="mean")
        total_loss += loss
        loss.backward()
        avg_loss = total_loss / i
        tp = output[target == 1].sum()
        precision = tp / output.sum()
        tn = (1 - output)[target == 0].sum()
        recall = tn.sum() / (1 - output).sum()
        tpr = tp / target.sum()
        tnr = tn / (1 - target).sum()
        optimizer.step()
        step = i + start
        if i % log_interval == 0:
            log_progress = tqdm(total=log_interval, desc="next log")
            writer.add_scalar("loss", loss, step)
            writer.add_scalar("avg_loss", avg_loss, step)
            writer.add_scalar("precision", precision, step)
            writer.add_scalar("recall", recall, step)
            writer.add_scalar("1-precision", 1 - precision, step)
            writer.add_scalar("1-recall", 1 - recall, step)
            writer.add_scalar("tpr", tpr, step)
            writer.add_scalar("tnr", tnr, step)
            writer.add_scalar("1-tpr", 1 - tpr, step)
            writer.add_scalar("1-tnr", 1 - tnr, step)
            writer.add_scalar("balanced_acc", (tpr + tnr) / 2, step)
            writer.add_scalar("1-balanced_acc", 1 - (tpr + tnr) / 2, step)

        if i % save_interval == 0:
            torch.save(network.state_dict(), str(Path(log_dir, "network.pt")))
        log_progress.update()


def maybe_int(string):
    if string == "None":
        return None
    return int(string)


def cli():
    # Training settings
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate "
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed ")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--log-dir", default="/tmp/mnist", metavar="N", help="")
    parser.add_argument("--run-id", default="", metavar="N", help="")
    parser.add_argument("--load-path", type=Path)
    dataset_parser = parser.add_argument_group("dataset_args")
    dataset_parser.add_argument("--lower-level-config", type=Path, required=True)
    dataset_parser.add_argument("--lower-level-load-path", type=Path, required=True)
    ppo.control_flow.multi_step.env.build_parser(
        dataset_parser,
        default_max_while_loops=2,
        default_max_world_resamples=0,
        default_min_lines=1,
        default_max_lines=20,
        default_time_to_waste=0,
    )
    network_parser = parser.add_argument_group("network_args")
    network_parser.add_argument(
        f"--pool-type",
        choices=("avg", "max", None),
        type=lambda s: None if s == "None" else s,
    )
    network_parser.add_argument(f"--concat", action="store_true")
    network_parser.add_argument(f"--line-hidden-size", type=int, required=True)
    network_parser.add_argument(f"--lower-hidden-size", type=int, required=True)
    for i in range(MAX_LAYERS):
        network_parser.add_argument(
            f"--conv-layer{i}", dest="conv_layers", action="append", type=maybe_int
        )
        for mod in ("conv", "pool"):
            for component in ("kernel", "stride"):
                network_parser.add_argument(
                    f"--{mod}-{component}{i}",
                    dest=f"{mod}_{component}s",
                    action="append",
                    type=maybe_int,
                )
    main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    cli()
