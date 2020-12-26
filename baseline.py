import hydra
from omegaconf import DictConfig

import baseline_agent
import ours
from wrappers import VecPyTorch


class Trainer(ours.Trainer):
    @staticmethod
    def build_agent(envs: VecPyTorch, **agent_args):
        return baseline_agent.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            **agent_args,
        )


@hydra.main(config_name="config")
def app(cfg: ours.OurConfig) -> None:
    cfg.eval.interval = cfg.eval.steps = None
    assert isinstance(cfg, DictConfig)
    Trainer.main(cfg)


if __name__ == "__main__":
    ours.main(app)
