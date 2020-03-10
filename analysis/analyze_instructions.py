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


def count(
    instruction: np.ndarray, successes: np.ndarray, line_type: L
) -> Generator[float, None, None]:
    yield np.sum(instruction[:, 0] == line_type.value - 1)


def main(root: Path, path: Path, line_types: List[L]) -> None:
    instruction_paths = list(Path(root, path).glob("**/eval_instruction.npz"))
    success_paths = list(Path(root, path).glob("**/eval_successes.npy"))
    assert len(instruction_paths) == len(success_paths)

    for line_type in line_types:

        def iterator() -> Generator[float, None, None]:
            for instruction_path, success_path in tqdm(
                zip(instruction_paths, success_paths), total=len(success_paths)
            ):
                try:
                    instructions = np.load(instruction_path)
                    successes = np.load(success_path)
                except zipfile.BadZipFile:
                    continue
                assert len(instructions) == len(successes)
                for args in zip(instructions.values(), successes):
                    yield from count(*args, line_type=line_type)  # type: ignore

        counts = np.array(list(iterator()))
        hist, _ = np.histogram(counts, bins=list(range(20)))
        print(line_type, *hist, sep=",")

    # _ = plt.hist(counts, bins=list(range(10)))  # arguments are passed to np.histogram
    # plt.savefig("counts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--line-types", type=lambda s: getattr(L, s), nargs="+", required=True
    )
    main(**vars(parser.parse_args()))
