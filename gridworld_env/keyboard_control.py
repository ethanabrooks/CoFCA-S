import argparse
import time

import gym


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=gym.make)
    parser.add_argument("seed", type=int)
    run(**vars(parser.parse_args()))


def run(env, actions, seed):
    env.seed(seed)
    actions = list(actions)

    s = env.reset()
    while True:
        env.render()
        action = None
        while action not in actions:
            action = input("act:")
            if action == "p":
                import ipdb

                ipdb.set_trace()

        s, r, t, i = env.step(actions.index(action))
        print("reward", r)
        if t:
            env.render()
            print("resetting")
            time.sleep(0.5)
            env.reset()
            print()


if __name__ == "__main__":
    # noinspection PyUnresolvedReferences
    cli()
