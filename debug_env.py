import env
from data_types import Building, Resource


class Env(env.Env):
    @staticmethod
    def building_allowed(building_positions, worker_position, *args, **kwargs):
        return worker_position not in building_positions


def main(**kwargs):
    Env(rank=0, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
