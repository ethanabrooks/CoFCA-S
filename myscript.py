from stable_baselines3.common.vec_env import SubprocVecEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv
from multiprocessing import Queue
from env import Env


def create_thunk():
    # return lambda: MyEnv(queue)
    return lambda: Env(
        break_on_fail=False,
        attack_prob=0,
        max_lines=10,
        min_lines=1,
        num_initial_buildings=2,
        time_per_line=4,
        tgt_success_rate=0.75,
        world_size=3,
        eval_steps=500,
        failure_buffer=queue,
        random_seed=0,
        rank=0,
    )


if __name__ == "__main__":
    queue = Queue()
    envs = SubprocVecEnv(
        env_fns=[create_thunk() for _ in range(2)], start_method="fork"
    )
    print(envs.reset())
