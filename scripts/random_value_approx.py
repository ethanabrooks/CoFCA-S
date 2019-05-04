import argparse
import sys
import configparser
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn


def get_distribution(arg):
    return dict(
        uniform=np.random.uniform,
        poisson=np.random.poisson,
        normal=np.random.normal)[arg]


def get_estimates(values, num_steps, num_seeds, exploration_bonus, noise_scale):
    u = np.ones(values.size)
    logits_list = [u.copy() for _ in range(num_seeds)]

    def sample(logits, adaptive):
        probs = logits / logits.sum()
        index = np.random.choice(values.size, p=probs)
        choice = values[index]
        if adaptive:
            logits += exploration_bonus
            logits[index] = np.abs(choice)
        return choice / (values.size * probs[index])

    for i in range(num_steps):
        true = values.mean()
        adaptive = [sample(l, adaptive=True) for l in logits_list]
        baseline = [sample(u, adaptive=False) for _ in logits_list]
        yield true, adaptive, baseline
        values += noise_scale * np.random.normal(1, 1, values.size)


def main(distribution, stats, num_values, num_steps, seed, num_samples,
         exploration_bonus, num_seeds, noise_scale, noise_mean, noise_std):
    if seed is not None:
        np.random.seed(seed)

    def get_estimates(values):
        u = np.ones(values.size)
        logits_list = [u.copy() for _ in range(num_seeds)]

        def sample(logits, adaptive):
            probs = logits / logits.sum()
            indices = np.random.choice(values.size, size=num_samples, p=probs)
            choice = values[indices]
            if adaptive:
                logits[indices] = np.abs(choice)
                logits += exploration_bonus
            weight = 1 / (values.size * probs[indices])
            return np.mean(choice * weight)

        for i in range(num_steps):
            for logits in logits_list:
                yield i, sample(logits, adaptive=True), 'adaptive'
                yield i, sample(u, adaptive=False), 'baseline'
            yield i, values.mean(), 'truth'
            values += noise_scale * np.random.normal(noise_mean, noise_std, values.size)

    estimates = get_estimates(distribution(*stats, size=num_values).astype(float))
    data = pd.DataFrame(data=estimates, columns=['steps', 'estimate', 'type'])

    seaborn.lineplot(x='steps', y='estimate', hue='type', data=data)
    plt.savefig('fig')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', type=get_distribution)
    parser.add_argument('--stats', nargs='*', type=float)
    parser.add_argument('--num-values', type=int, default=20)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exploration-bonus', type=float, default=.01)
    parser.add_argument('--num-seeds', type=int, default=100)
    parser.add_argument('--noise-scale', type=float, default=.01)
    parser.add_argument('--noise-std', type=float, default=1)
    parser.add_argument('--noise-mean', type=float, default=1)
    parser.add_argument('--num-samples', type=float, default=1)


    if len(sys.argv) == 2:
        import ipdb; ipdb.set_trace()
        with open(sys.argv.pop(1)) as f:
            parser.set_defaults(**json.load(f))


    main(**vars(parser.parse_args()))
