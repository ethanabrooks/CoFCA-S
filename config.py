from collections import namedtuple

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

Parsers = namedtuple("Parser", "main agent ppo rollouts")


from dataclasses import dataclass
from typing import Optional


def flatten(cfg: DictConfig):
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            for k_, v_ in flatten(v):
                yield f"{k}_{k_}", v_
        else:
            yield k, v


@dataclass
class BaseConfig:
    activation_name: str = "ReLU"
    clip_param: float = 0.2
    cuda_deterministic: bool = True
    entropy_coef: float = 0.25
    eval_interval: Optional[int] = None
    gamma: float = 0.99
    group: Optional[str] = None
    hidden_size: int = 150
    learning_rate: float = 0.0025
    load_path: Optional[str] = None
    log_interval: int = int(1e5)
    max_grad_norm: float = 0.5
    name: Optional[str] = None
    normalize: bool = False
    num_batch: int = 1
    num_processes: int = 100
    optimizer: str = "Adam"
    ppo_epoch: int = 5
    cuda: bool = True
    use_wandb: bool = True
    num_frames: Optional[int] = None
    render: bool = False
    render_eval: bool = False
    save_interval: int = int(1e5)
    seed: int = 0
    synchronous: bool = False
    tau: float = 0.95
    train_steps: int = 25
    use_gae: bool = False
    value_loss_coef: float = 0.5
    wandb_version: Optional[str] = None
    _wandb: Optional[str] = None


@dataclass
class Config(BaseConfig):
    env: str = "CartPole-v0"
    num_layers: int = 100
    recurrent: bool = False


cs = ConfigStore.instance()
