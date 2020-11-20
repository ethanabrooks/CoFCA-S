import env


class Env(env.Env):
    @staticmethod
    def building_allowed(*args, **kwargs):
        return True


def main(**kwargs):
    Env(rank=0, **kwargs).main()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    Env.add_arguments(PARSER)
    PARSER.add_argument("--seed", default=0, type=int)
    main(**vars(PARSER.parse_args()))
