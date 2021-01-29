from pprint import pprint

import hydra
from omegaconf import DictConfig

import unstructured_memory
import main
from wrappers import VecPyTorch


class Trainer(main.Trainer):
    @staticmethod
    def build_agent(envs: VecPyTorch, **agent_args):
        agent_args.update(feed_m_to_gru=False, globalized_critic=False)
        return unstructured_memory.Agent(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            **agent_args,
        )


@hydra.main(config_name="config")
def app(cfg: DictConfig) -> None:
    pprint(dict(**cfg))
    Trainer.main(cfg)


if __name__ == "__main__":
    main.main(app)
