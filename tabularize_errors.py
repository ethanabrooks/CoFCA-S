#! /usr/bin/env python

# stdlib
import argparse
from collections import deque, defaultdict
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
    parser.add_argument("--tags", nargs="*", help=" ")
    parser.add_argument("--until-time", type=int, help=" ")
    parser.add_argument("--until-step", type=int, help=" ")
    parser.add_argument("--fname", type=str, default="plot")
    parser.add_argument("--quality", type=int)
    parser.add_argument("--dpi", type=int, default=256)
    main(**vars(parser.parse_args()))


def tag_to_header(tag):
    if tag == "eval_failed_to_keep_up":
        return "fell behind"
    if tag == "eval_mistaken_id":
        return "wrong subtask"
    if tag == "eval_mistakenly_advanced":
        return "got ahead"
    return tag.lstrip("eval_").replace("_", " ")


def main(
    path: Path,
    tags: List[str],
    smoothing: int,
    until_time: int,
    until_step: int,
    **kwargs
):
    def get_values_from_path(event_path):
        values = defaultdict(lambda: deque(maxlen=smoothing))
        iterator = tensorflow.compat.v1.train.summary_iterator(str(event_path))
        for _ in itertools.count():
            try:
                event = next(iterator)
                start_time = event.wall_time
                for value in event.summary.value:
                    if value.tag in tags:
                        values[value.tag].append(value.simple_value)
                if until_time is not None and event.wall_time - start_time > until_time:
                    return values
                if until_step is not None and event.step > until_step:
                    return values
            except DataLossError:
                print("Data loss in", path)
            except StopIteration:
                return values

    def avg(x):
        return sum(x) / len(x)

    value_dict = defaultdict(list)
    for event_path in path.glob("**/events*"):
        for tag, values in get_values_from_path(event_path).items():
            if len(values) > 0:
                value_dict[tag].append(avg(values))

    for tag, values in value_dict.items():
        print(tag_to_header(tag), avg(values), sep=",")


if __name__ == "__main__":
    cli()
