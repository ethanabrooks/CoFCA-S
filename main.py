import argparse

from common.vec_env.util import hierarchical_parse_args
from arguments import add_arguments
from trainer import Trainer

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    Trainer().train(**hierarchical_parse_args(PARSER))
