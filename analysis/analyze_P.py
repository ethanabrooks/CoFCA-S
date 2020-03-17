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
from instruction_analysis import L


def analyze_P(
    instruction: np.ndarray, P: np.ndarray, start, stop
) -> Generator[int, None, None]:
    import ipdb  # type:ignore

    ipdb.set_trace()

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
    instruction_paths: Iterable[Path],
    P_paths: Iterable[Path],
    pairs: Iterable[Tuple[L, L]],
) -> Generator[Tuple[str, List[int]], None, None]:
    for start, stop in pairs:
        print(start)

        def iterator() -> Generator[int, None, None]:
            for instruction_path, P_path in tqdm(list(zip(instruction_paths, P_paths))):
                try:
                    instructions = np.load(instruction_path)
                    Ps = np.load(P_path)
                except zipfile.BadZipFile as e:
                    print(e)
                    continue
                assert len(instructions) == len(Ps)
                for instruction, P in zip(instructions.values(), Ps.values()):
                    yield from analyze_P(instruction, P, start, stop)

        # counts = np.array(list(iterator()))
        # hist, _ = np.histogram(counts, bins=list(range(20)))
        yield f"{start.name} length", list(iterator())


def main(root: Path, path: Path, evaluation: bool, **kwargs) -> None:
    def get_paths(filename):
        if evaluation:
            filename = "eval_" + filename
        return Path(root, path).glob("**/" + filename)

    instruction_paths = list(get_paths("instruction.npz"))
    P_paths = list(get_paths("P.npz"))
    assert len(instruction_paths) == len(P_paths)
    names, lists = zip(*list(generate_iterators(instruction_paths, P_paths, **kwargs)))
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
    parser.add_argument("--evaluation", action="store_true")
    main(**vars(parser.parse_args()))
