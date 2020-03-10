#! /usr/bin/env python
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum
from typing import Dict, Tuple, List, Generator, Optional
from tqdm import tqdm  # type: ignore

# import matplotlib.pyplot as plt
import zipfile


L = Enum("line", "If Else EndIf While EndWhile EndLoop Subtask Padding Loop Any")


def count(instruction: np.ndarray, start, stop) -> Generator[int, None, None]:
    num_steps = None
    for i in instruction[:, 0]:
        if num_steps is not None:
            num_steps += 1
        if i == start.value - 1:
            num_steps = 0
        if i == stop.value - 1:
            yield num_steps


def main(root: Path, path: Path, evaluation: bool, pairs) -> None:
    filename = "instruction.npz"
    if evaluation:
        filename = "eval_" + filename
    instruction_paths = list(Path(root, path).glob("**/" + filename))

    for start, stop in pairs:

        def iterator() -> Generator[float, None, None]:
            for instruction_path in tqdm(
                instruction_paths, total=len(instruction_paths)
            ):
                try:
                    instructions = np.load(instruction_path)
                except zipfile.BadZipFile:
                    continue
                for instruction in instructions.values():
                    yield from count(instruction, start, stop)

        counts = np.array(list(iterator()))
        hist, _ = np.histogram(counts, bins=list(range(20)))
        print(start, *hist, sep=",")

    # _ = plt.hist(counts, bins=list(range(10)))  # arguments are passed to np.histogram
    # plt.savefig("counts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--pair", nargs=2, dest="pairs", type=lambda s: getattr(L, s), action="append"
    )
    parser.add_argument("--evaluation", action="store_true")
    main(**vars(parser.parse_args()))
