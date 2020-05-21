#! /usr/bin/env python
import argparse
import csv
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


def generate_lengths(
    instruction_paths: Iterable[Path],
    success_paths: Iterable[Path],
    line_types: Iterable[L],
    pairs: Iterable[Tuple[L, L]],
) -> Generator[Tuple[str, List[int]], None, None]:
    # for instruction_path, success_path in tqdm(zip(instruction_paths, success_paths)):
    # try:
    # instructions = np.load(instruction_path)
    # successes = np.load(success_path)
    # except zipfile.BadZipFile:
    # continue
    # assert len(instructions) == len(successes)
    # for instruction, success in zip(instructions.values(), successes):
    # yield [success] + [
    # count(instruction, line_type) for line_type in line_types
    # ]

    for start, stop in pairs:
        for instruction_path, success_path in tqdm(
            zip(instruction_paths, success_paths)
        ):
            try:
                instructions = np.load(instruction_path)
                successes = np.load(success_path)
            except zipfile.BadZipFile:
                continue
            assert len(instructions) == len(successes)
            for i, (instruction, success) in enumerate(
                zip(instructions.values(), successes)
            ):
                # yield from measure_length(instruction, start, stop)
                for length in measure_length(instruction, start, stop):
                    if length is not None:
                        yield success, L(start).name, L(stop).name, i, length

        # yield name, list(iterator())


def main(
    root: Path, path: Path, training: bool, out: Path, line_types: Iterable[L], **kwargs
) -> None:
    instruction_filename = "instruction.npz"
    success_filename = "successes.npy"
    if not training:
        instruction_filename = "eval_" + instruction_filename
        success_filename = "eval_" + success_filename
    instruction_paths = list(Path(root, path).glob("**/" + instruction_filename))
    success_paths = list(Path(root, path).glob("**/" + success_filename))
    with out.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["success", "start", "end", "episode", "length"])
        for row in generate_lengths(
            instruction_paths=instruction_paths,
            success_paths=success_paths,
            line_types=line_types,
            **kwargs,
        ):
            print(row)
            writer.writerow(row)

    # names, lists = zip(
    # *list(generate_iterators(instruction_paths, success_paths, **kwargs))
    # )
    # for name, array in zip(names, map(np.array, lists)):
    # hist, _ = np.histogram(array, bins=list(range(20)))
    # print(name, *hist, sep=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pair", nargs=2, dest="pairs", type=lambda s: getattr(L, s), action="append"
    )
    parser.add_argument(
        "--line-types",
        type=lambda s: getattr(L, s),
        nargs="+",
        default=[l.value for l in L],
    )
    parser.add_argument("--training", action="store_true")
    main(**vars(parser.parse_args()))
