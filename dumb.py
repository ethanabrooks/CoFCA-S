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


@dataclass
class Search:
    values: List[Any] = MISSING


@dataclass
class BaseAgentConfig:
    entropy_coef: float = 0.25
    hidden_size: int = 150


@dataclass
class AgentConfig(BaseAgentConfig):
    num_layers: int = 100
    recurrent: bool = False


@dataclass
class EnvConfig:
    env: str = "CartPole-v0"


@dataclass
class Config:
    agent_config: Any = MISSING
    clip_param: float = 0.2
    cuda_deterministic: bool = True
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
class OurConfig(Config):
    curriculum_level: int = 0
    curriculum_threshold: float = 0.9
    debug_env: bool = False
    no_eval: bool = True
    failure_buffer_load_path: Optional[str] = None
    failure_buffer_size: int = 10000
    max_eval_lines: int = 50
    min_eval_lines: int = 1
    conv_hidden_size: int = 100
    debug: bool = True
    gate_coef: float = 0.01
    resource_hidden_size: int = 128
    kernel_size: int = 2
    lower_embed_size: int = 75
    max_lines: int = 10
    min_lines: int = 1
    next_actions_embed_size: int = 25
    num_edges: int = 1
    no_pointer: bool = False
    no_roll: bool = False
    no_scan: bool = False
    olsk: bool = False
    stride: int = 1
    task_embed_size: int = 128
    transformer: bool = False


@dataclass
class EnvConfig(AgentConfig):
    assimilator_prob: float = 0.5
    break_on_fail: bool = False
    destroy_building_prob: float = 0
    num_initial_buildings: int = 0
    time_per_line: int = 4
    tgt_success_rate: float = 0.75
    world_size: int = 3


cs = ConfigStore.instance()
cs.store(name="config", node=OurConfig)
cs.store(group="eval_config", name="no_eval", node=NoEval)
cs.store(group="eval_config", name="yes_eval", node=YesEval)
cs.store(group="agent_config", name="agent", node=AgentConfig)
cs.store(group="agent_config", name="our_agent", node=BaseAgentConfig)


@hydra.main(config_name="config")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
#
