import argparse

from common.vec_env.util import hierarchical_parse_args
from ppo.arguments import add_arguments
from ppo.train import Train

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    Train(**hierarchical_parse_args(PARSER)).run()
