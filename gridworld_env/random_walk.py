import argparse
import time

import gym


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=gym.make)
    run(**vars(parser.parse_args()))


def run(env):
    env.reset()
    while True:
        env.render()
        # time.sleep(.5)
        s, r, t, i = env.step(env.action_space.sample())
        print('reward', r)
        if t:
            env.render()
            print('resetting')
            # time.sleep(1)
            env.reset()
            print()


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import gridworld_env
    cli()
