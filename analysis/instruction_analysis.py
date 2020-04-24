#! /usr/bin/env python
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum, auto
from typing import Dict, Tuple, Generator, Optional, Iterable, List
from tqdm import tqdm  # type: ignore
import zipfile


class L(Enum):
    Subtask = 0
    If = 1
    Else = 2
    EndIf = 3
    While = 4
    EndWhile = 5
    Loop = 6
    EndLoop = 7
    Padding = 8
    Any = 9

    def __eq__(self, i) -> bool:
        return i == self.value


def count(instruction: np.ndarray, line_type: L) -> Generator[int, None, None]:
    yield int(np.sum(line_type == instruction[:, 0]))


def measure_length(instruction: np.ndarray, start, stop) -> Generator[int, None, None]:
    def line_type_generator():
        yield from instruction[:, 0]

    def go_to_stop(it):
        for j, i in enumerate(it):
            if stop == i:
                return j

    def go_to_start(it):
        for i in it:
            if start == i:
                return True
        return False

    num_steps = None
    line_type_iterator = line_type_generator()
    while go_to_start(line_type_iterator):
        yield go_to_stop(line_type_iterator)


def generate_iterators(
    paths: Iterable[Path], line_types: Iterable[L], pairs: Iterable[Tuple[L, L]]
) -> Generator[Tuple[str, List[int]], None, None]:
    for line_type in line_types:
        print(line_type.name)

        def iterator() -> Generator[int, None, None]:
            for instruction_path in tqdm(paths):
                try:
                    instructions = np.load(instruction_path)
                    # successes = np.load(success_path)
                except zipfile.BadZipFile:
                    continue
                # assert len(instructions) == len(successes)
                for instruction in instructions.values():
                    yield from count(instruction, line_type)

        yield line_type.name, list(iterator())

    for start, stop in pairs:
        name = f"{start.name}-{stop.name} length"
        print(name)

        def iterator() -> Generator[int, None, None]:
            for instruction_path in tqdm(paths):
                try:
                    instructions = np.load(instruction_path)
                except zipfile.BadZipFile:
                    continue
                for instruction in instructions.values():
                    # yield from measure_length(instruction, start, stop)
                    for length in measure_length(instruction, start, stop):
                        if length is not None:
                            yield length

        yield name, list(iterator())


def main(root: Path, path: Path, evaluation: bool, **kwargs) -> None:
    filename = "instruction.npz"
    if evaluation:
        filename = "eval_" + filename
    instruction_paths = list(Path(root, path).glob("**/" + filename))
    names, lists = zip(*list(generate_iterators(instruction_paths, **kwargs)))
    for name, array in zip(names, map(np.array, lists)):
        hist, _ = np.histogram(array, bins=list(range(20)))
        print(name, *hist, sep=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--pair", nargs=2, dest="pairs", type=lambda s: getattr(L, s), action="append"
    )
    parser.add_argument(
        "--line-types", type=lambda s: getattr(L, s), nargs="+", required=True
    )
    parser.add_argument("--evaluation", action="store_true")
    main(**vars(parser.parse_args()))
