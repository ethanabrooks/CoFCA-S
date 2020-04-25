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
    def line_type_generator():
        yield from instruction[:, 0]

    def go_to(it, val):
        for j, i in enumerate(it):
            if val == i:
                return j

    num_steps = None
    line_type_iterator = line_type_generator()
    while True:
        i = go_to(line_type_iterator, start)
        if i is None:
            break
        half = len(P) - 1
        ex = np.arange(len(P[i])) @ P[i] - half
        delta = go_to(line_type_iterator, stop)
        if delta is not None:
            yield min(ex, key=lambda x: abs(x - delta))


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
                    if not P.shape[:2] == (1, 1):
                        import ipdb

                        ipdb.set_trace()
                    yield from analyze_P(
                        instruction, np.squeeze(P, axis=(0, 1)), start, stop
                    )

        # counts = np.array(list(iterator()))
        # hist, _ = np.histogram(counts, bins=list(range(20)))
        yield f"{start.name}-{stop.name} edge", list(iterator())


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
        counts, bins = np.histogram(array)
        print(f"{name} bins", *bins, sep=",")
        print(f"{name} counts", *counts, sep=",")
    for name, array in zip(names, map(np.array, lists)):
        print(f"{name} mean", np.mean(array))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--pair", nargs=2, dest="pairs", type=lambda s: getattr(L, s), action="append"
    )
    parser.add_argument("--evaluation", action="store_true")
    main(**vars(parser.parse_args()))
