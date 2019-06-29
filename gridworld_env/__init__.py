import json
from pathlib import Path

from gym.envs import register

from gridworld_env.gridworld import GridWorld
from gridworld_env.random_gridworld import RandomGridWorld
from gridworld_env.simple_pomdp import SimplePOMDP
from gridworld_env.subtasks_gridworld import SubtasksGridWorld  # noqa

SUFFIX = "GridWorld-v0"
JSON_PATH = Path(__file__).parent.joinpath("json")


def register_from_string(env_id, class_=None, **kwargs):
    if class_ == "SubtasksGridWorld":
        class_ = SubtasksGridWorld
    elif "random" in kwargs:
        class_ = RandomGridWorld
    else:
        class_ = GridWorld

    register(
        id=env_id,
        entry_point=f"{class_.__module__}:{class_.__name__}",
        reward_threshold=kwargs.pop("reward_threshold", None),
        max_episode_steps=kwargs.pop("max_episode_steps", None),
        nondeterministic=False,
        kwargs=kwargs,
    )


def get_args(env_id):
    assert env_id.endswith(SUFFIX)
    path = Path(JSON_PATH, env_id[: -len(SUFFIX)]).with_suffix(".json")
    with path.open("rb") as f:
        return dict(**json.load(f))


def register_envs():
    for path in JSON_PATH.iterdir():
        with path.open() as f:
            register_from_string(f"{path.stem}{SUFFIX}", **json.load(f))


register_envs()
entry_point = f"{SimplePOMDP.__module__}:{SimplePOMDP.__name__}"
register(
    id="POMDP-v0",
    entry_point=entry_point,
    max_episode_steps=SimplePOMDP.max_episode_steps,
)
