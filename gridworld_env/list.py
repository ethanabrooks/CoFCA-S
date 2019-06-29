from gridworld_env import JSON_PATH, SUFFIX


def cli():
    for path in JSON_PATH.iterdir():
        print(f"{path.stem}{SUFFIX}")
