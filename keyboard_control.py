import argparse
import time

import gym


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=gym.make)
    parser.add_argument("seed", type=int)
    run(**vars(parser.parse_args()))


def run(env, action_fn):
    s = env.reset()
    while True:
        env.render(pause=False)
        action = None
        while action is None:
            action = action_fn(input("act:"))
            if action == "p":
                import ipdb

                ipdb.set_trace()

        s, r, t, i = env.step(action)
        print("reward", r)
        if t:
            env.render(pause=False)
            print("resetting")
            time.sleep(0.5)
            env.reset()
            print()


if __name__ == "__main__":
    # noinspection PyUnresolvedReferences
    cli()
