from collections import namedtuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

Parsers = namedtuple("Parser", "main agent ppo rollouts")


from dataclasses import dataclass, field
from typing import Optional, Any, List


def flatten(cfg: DictConfig):
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            yield from flatten(v)
        else:
            yield k, v


@dataclass
class Eval:
    eval_interval: Optional[int] = MISSING
    eval_steps: Optional[int] = MISSING


@dataclass
class NoEval(Eval):
    eval_interval: Optional[int] = None
    eval_steps: Optional[int] = None


@dataclass
class YesEval(Eval):
    eval_interval: Optional[int] = int(1e6)
    eval_steps: Optional[int] = 500


@dataclass
class BaseConfig:
    clip_param: float = 0.2
    config: Optional[str] = None
    cuda_deterministic: bool = True
    entropy_coef: float = 0.25
    eval: Any = MISSING
    eps: float = 1e-5
    gamma: float = 0.99
    group: Optional[str] = None
    hidden_size: int = 150
    learning_rate: float = 0.0025
    load_path: Optional[str] = None
    log_interval: int = int(2e4)
    max_grad_norm: float = 0.5
    name: Optional[str] = None
    perform_eval: bool = False
    cuda: bool = True
    use_wandb: bool = True
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
    defaults: List[Any] = field(default_factory=lambda: [dict(eval="eval")])


@dataclass
class Config(BaseConfig):
    env: str = "CartPole-v0"
    num_layers: int = 100
    recurrent: bool = False


cs = ConfigStore.instance()
cs.store(group="eval", name="eval", node=NoEval)
