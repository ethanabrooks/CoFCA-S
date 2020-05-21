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
from lengths import L
import csv


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
            yield delta, ex


def generate_offsets(
    instruction_paths: Iterable[Path],
    P_paths: Iterable[Path],
    success_paths: Iterable[Path],
    pairs: Iterable[Tuple[L, L]],
) -> Generator[Tuple[str, List[int]], None, None]:
    for start, stop in pairs:
        for instruction_path, P_path, success_path in tqdm(
            list(zip(instruction_paths, P_paths, success_paths))
        ):
            try:
                instructions = np.load(instruction_path)
                Ps = np.load(P_path)
                successes = np.load(success_path)
            except zipfile.BadZipFile as e:
                print(e)
                continue
            assert len(instructions) == len(Ps)
            for i, (instruction, P, success) in enumerate(
                zip(instructions.values(), Ps.values(), successes)
            ):
                # if not P.shape[:2] == (1, 1):
                # import ipdb

                # ipdb.set_trace()
                for d, x in analyze_P(
                    instruction, np.squeeze(P, axis=(0,)), start, stop
                ):
                    yield (success, L(start).name, L(stop).name, i, d, *x)


def main(root: Path, path: Path, out: Path, evaluation: bool, **kwargs) -> None:
    def get_paths(filename):
        if evaluation:
            filename = "eval_" + filename
        return Path(root, path).glob("**/" + filename)

    instruction_paths = list(get_paths("instruction.npz"))
    success_paths = list(get_paths("successes.npy"))
    P_paths = list(get_paths("P.npz"))
    assert len(instruction_paths) == len(P_paths)
    assert len(instruction_paths) == len(success_paths)
    with out.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["success", "start", "end", "episode", "length", "learned edge"]
        )
        for row in generate_offsets(
            instruction_paths=instruction_paths,
            P_paths=P_paths,
            success_paths=success_paths,
            **kwargs
        ):
            print(row)
            writer.writerow(row)
    # names, lists = zip(
    # *list(generate_iterators(instruction_paths, P_paths, success_paths, **kwargs))
    # )
    # for name, array in zip(names, map(np.array, lists)):
    # counts, bins = np.histogram(array)
    # print(f"{name} bins", *bins, sep=",")
    # print(f"{name} counts", *counts, sep=",")
    # for name, array in zip(names, map(np.array, lists)):
    # print(f"{name} mean", np.mean(array))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pair", nargs=2, dest="pairs", type=lambda s: getattr(L, s), action="append"
    )
    parser.add_argument("--evaluation", action="store_true")
    main(**vars(parser.parse_args()))
