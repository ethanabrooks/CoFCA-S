#! /usr/bin/env python

# stdlib
import argparse
from pathlib import Path
from typing import List, Optional

# third party
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import re


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--names", nargs="*", type=Path)
    parser.add_argument("--paths", nargs="*", type=Path)
    parser.add_argument("--base-dir", default=".runs/logdir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fname", type=str, default="plot")
    parser.add_argument("--quality", type=int)
    parser.add_argument("--dpi", type=int, default=256)
    main(**vars(parser.parse_args()))


def main(
    names: List[str],
    paths: List[Path],
    base_dir: Path,
    limit: Optional[int],
    quiet: bool,
    **kwargs,
):
    if not len(names) == len(paths):
        raise RuntimeError(
            f"These values should have the same number of values:\nnames: ({names})\npaths: ({paths}))"
        )

    def get_tags():
        pattern = re.compile(r".*cumulative_reward_([0-9]*)")
        for name, path in zip(names, paths):
            path = Path(base_dir, path)
            if not path.exists():
                if not quiet:
                    print(f"{path} does not exist")

            for event_path in path.glob("**/events*"):
                print("Reading", event_path)
                iterator = tf.compat.v1.train.summary_iterator(str(event_path))

                while True:
                    try:
                        event = next(iterator)
                        value = event.summary.value
                        if value:
                            tag = value[0].tag
                            match = pattern.match(tag)
                            if match:
                                value = value[0].simple_value
                                if limit is None or event.step < limit:
                                    jump = int(match.groups()[0])
                                    yield jump, value, name
                    except DataLossError:
                        pass
                    except StopIteration:
                        break

    print("Plotting...")
    data = pd.DataFrame(get_tags(), columns=["jump", "reward", "run"]).sort_values(
        "jump"
    )
    sns.lineplot(x="jump", y="reward", hue="run", data=data)
    plt.legend(data["run"].unique())
    # plt.axes().ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    plt.savefig(**kwargs)


if __name__ == "__main__":
    cli()
