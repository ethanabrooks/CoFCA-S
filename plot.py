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
    parser.add_argument("--font-size", type=int)
    parser.add_argument("--fname", type=str, default="plot")
    parser.add_argument("--quality", type=int)
    parser.add_argument("--dpi", type=int, default=256)
    main(**vars(parser.parse_args()))


def main(
    names: List[str],
    font_size: int,
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
        pattern = re.compile(r"eval_cumulative_reward_([0-9]*)")
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
    data = pd.DataFrame(get_tags(), columns=["block size", "reward", "run"])
    ax = sns.lineplot(x="block size", y="reward", hue="run", data=data)
    ax.set_xlabel("block size", fontsize=font_size)
    ax.set_ylabel("success rate", fontsize=font_size)
    ax.axes.get_xaxis().get_label().set_visible(False)
    ax.tick_params(labelsize=font_size)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], fontsize=font_size - 2)
    plt.tight_layout()
    # plt.axes().ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    plt.savefig(**kwargs, bbox_inches="tight")


if __name__ == "__main__":
    cli()
