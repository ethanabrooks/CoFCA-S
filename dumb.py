from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


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


defaults = [dict(eval_config="no_eval"), dict(agent="agent")]


@dataclass
class Search:
    values: List[Any] = MISSING


@dataclass
class AgentConfig:
    entropy_coef: float = 0.25
    hidden_size: int = 150
    num_layers: int = 100
    recurrent: bool = False


@dataclass
class OurAgentConfig(AgentConfig):
    train_steps = 20
    hello: int = 6


@dataclass
class Config:
    agent: Any = MISSING
    clip_param: float = 0.2
    cuda_deterministic: bool = True
    env: str = "CartPole-v0"
    eps: float = 1e-5
    eval_config: Any = MISSING
    gamma: float = 0.99
    group: Optional[str] = None
    learning_rate: float = 0.0025
    load_path: Optional[str] = None
    log_interval: int = int(2e4)
    max_grad_norm: float = 0.5
    no_cuda: bool = False
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
    threshold: Optional[float] = None
    use_gae: bool = False
    value_loss_coef: float = 0.5

    defaults: List[Any] = field(default_factory=lambda: defaults)


@dataclass
class OurConfig(AgentConfig):
    curriculum_level: int = 0
    curriculum_threshold: float = 0.9
    debug_env: bool = False
    no_eval: bool = True
    failure_buffer_load_path: Optional[str] = None
    failure_buffer_size: int = 10000


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="eval_config", name="no_eval", node=NoEval)
cs.store(group="eval_config", name="yes_eval", node=YesEval)
cs.store(group="agent", name="agent", node=AgentConfig)
cs.store(group="agent", name="agent", node=OurAgentConfig)


@hydra.main(config_name="config")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
#
