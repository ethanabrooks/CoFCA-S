import socket

import ray
import torch
from ray.tune.result import TIME_TOTAL_S
from ray.tune.schedulers import AsyncHyperBandScheduler


def tune_main(debug, redis_port, log_dir, num_samples, tune_metric, **kwargs):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    ray.init(redis_address=f"{ip}:{redis_port}", local_mode=debug)
    kwargs.update(
        use_gae=ray.tune.choice([True, False]),
        num_batch=ray.tune.choice([1]),
        num_steps=ray.tune.choice([16, 32, 64, 128]),
        seed=ray.tune.choice(list(range(10))),
        entropy_coef=ray.tune.uniform(low=0.01, high=0.04),
        hidden_size=ray.tune.choice([32, 64, 128, 256]),
        num_layers=ray.tune.choice([0, 1, 2]),
        learning_rate=ray.tune.uniform(low=0.0001, high=0.005),
        ppo_epoch=ray.tune.choice(list(range(1, 6))),
        # feed_r_initially=ray.tune.choice([True, False]),
        # use_M_plus_minus=ray.tune.choice([True, False]),
        num_processes=300,
        # ppo_epoch=ray.tune.sample_from(
        #     lambda spec: max(
        #         1,
        #         int(
        #             1.84240459e06 * spec.config.learning_rate ** 2
        #             - 1.14376715e04 * spec.config.learning_rate
        #             + 1.89339209e01
        #             + np.random.uniform(low=-2, high=2)
        #         ),
        #     )
        # ),
    )

    class _Train(TrainControlFlow, ray.tune.Trainable):
        def _setup(self, config):
            def setup(
                agent_args,
                ppo_args,
                entropy_coef,
                hidden_size,
                num_layers,
                learning_rate,
                ppo_epoch,
                **kwargs,
            ):
                agent_args.update(
                    entropy_coef=entropy_coef,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )
                ppo_args.update(ppo_epoch=ppo_epoch, learning_rate=learning_rate)
                self.setup(**kwargs, agent_args=agent_args, ppo_args=ppo_args)

            setup(**config)

        def get_device(self):
            return "cuda"

    ray.tune.run(
        _Train,
        config=kwargs,
        resources_per_trial=dict(cpu=1, gpu=0.5 if torch.cuda.is_available() else 0),
        checkpoint_freq=1,
        reuse_actors=True,
        num_samples=1 if debug else num_samples,
        local_dir=log_dir,
        scheduler=AsyncHyperBandScheduler(
            time_attr=TIME_TOTAL_S,
            metric=tune_metric,
            mode="max",
            grace_period=3600,
            max_t=43200,
        ),
    )
