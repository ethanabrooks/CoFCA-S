from collections import namedtuple

from hydra.core.config_store import ConfigStore

Parsers = namedtuple("Parser", "main agent ppo rollouts")


from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    clip_param: float = 0.2
    cuda_deterministic: bool = True
    entropy_coef: float = 0.25
    eps: float = 1e-5
    eval_interval: Optional[int] = None
    eval_steps: Optional[int] = None
    gamma: float = 0.99
    group: Optional[str] = None
    hidden_size: int = 150
    learning_rate: float = 0.0025
    load_path: Optional[str] = None
    log_interval: int = int(2e4)
    max_grad_norm: float = 0.5
    name: Optional[str] = None
    no_eval: bool = True
    cuda: bool = True
    no_wandb: bool = False
    normalize: bool = False
    num_batch: int = 1
    num_frames: int = int(1e8)
    num_processes: int = 100
    ppo_epoch: int = 5
    render: bool = False
    render_eval: bool = False
    save_interval: int = int(2e4)
    seed: int = 0
    synchronous: bool = False
    tau: float = 0.95
    train_steps: int = 25
    use_gae: bool = False
    value_loss_coef: float = 0.5


@dataclass
class Config(BaseConfig):
    env: str = "CartPole-v0"
    num_layers: int = 100
    recurrent: bool = False
