#! /usr/bin/env python
from pathlib import Path

from ppo.main import exp_main
import socket

if __name__ == '__main__':

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    exp_main(
        cuda_deterministic=True,
        log_dir=str(Path("~/tune_results")),
        run_id="tune",
        num_processes=300,
        redis_address=f'{ip}:6379',
        tune=True,
        quiet=True,
        time_limit=30,
        eval_interval=300,
        log_interval=10,
        save_interval=300,
        env='',
        baseline=False,
        debug=False,
        single_instruction=False,
        oh_et_al=False,
        num_samples=100,
        num_batch=-1,
        num_steps=-1,
        seed=-1,
        entropy_coef=-1,
        hidden_size=-1,
        num_layers=-1,
        learning_rate=-1,
        ppo_epoch=-1,
        gridworld_args=dict(
            height=4,
            width=4,
            cook_time=2,
            time_to_heat_oven=3,
            doorbell_prob=0.05,
            mouse_prob=0.2,
            baby_prob=0.1,
            mess_prob=0.01,
            fly_prob=0.005,
            toward_cat_prob=0.5
        ),
        wrapper_args=dict(
            n_active_instructions=2,
            vision_range=1,
            watch_baby_range=2,
            avoid_dog_range=2,
            max_time_outside=15,
            test=["WatchBaby", "KillFlies"],
            valid=[],
            instructions=["AnswerDoor", "AvoidDog", "ComfortBaby", "KillFlies", "MakeFire", "WatchBaby"],
        )
    )
