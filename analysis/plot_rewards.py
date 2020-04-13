#! /usr/bin/env python

# stdlib
import argparse
from collections import Counter, deque
import itertools
from pathlib import Path
from typing import List

# third party
from tensorflow.python.framework.errors_impl import DataLossError
import tensorflow.compat.v1.train
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--smoothing", type=int, default=10, help=" ")
    parser.add_argument("--tag", default="rewards", help=" ")
    parser.add_argument("--until-time", type=int, help=" ")
    parser.add_argument("--until-step", type=int, help=" ")
    parser.add_argument("--fname", type=str, default="plot")
    parser.add_argument("--quality", type=int)
    parser.add_argument("--dpi", type=int, default=256)
    main(**vars(parser.parse_args()))


def main(
    path: Path, tag: str, smoothing: int, until_time: int, until_step: int, **kwargs
):
    def get_value_from_path(event_path):
        values = deque(maxlen=smoothing)
        max_line = int(event_path.parts[-3])

        def get_value_from_event(event):
            for value in event.summary.value:
                if value.tag == tag:
                    values.append(value.simple_value)

        iterator = tensorflow.compat.v1.train.summary_iterator(str(event_path))
        for _ in itertools.count():
            try:
                event = next(iterator)
                start_time = event.wall_time
                get_value_from_event(event)
                if until_time is not None and event.wall_time - start_time > until_time:
                    return sum(values) / smoothing
                if until_step is not None and event.step > until_step:
                    return sum(values) / smoothing
            except DataLossError:
                print("Data loss in", path)
            except StopIteration:
                if len(values) > 0:
                    return sum(values) / len(values)

    counter = Counter({"max line": [], "reward": []})
    for event_path in path.glob("**/events*"):
        max_line = int(event_path.parts[-3])
        reward = get_value_from_path(event_path)
        if reward is not None:
            counter.update({"max line": [max_line], "reward": [reward]})

    data = pd.DataFrame.from_dict(counter)
    sns.set()
    g = sns.catplot(x="max line", y="reward", kind="swarm", data=data)
    # g.add_legend(title="Max Lines per Instruction")

    # plt.axes().ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    g.savefig(**kwargs)


if __name__ == "__main__":
    cli()
