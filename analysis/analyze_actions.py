#! /usr/bin/env python
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum
from typing import Dict, Tuple, List, Generator, Optional
from tqdm import tqdm  # type: ignore

L = Enum("line", "If Else EndIf While EndWhile EndLoop Subtask Padding Loop Any")
EDGES: Dict[L, Tuple[List[L], List[L]]] = {
    L.If: ([], [L.Any, L.Else, L.EndIf]),
    L.Else: ([], [L.Any, L.EndIf]),
    L.EndIf: ([], [L.Any]),
    L.While: ([], [L.Any, L.EndWhile]),
    L.EndWhile: ([L.While], [L.Any]),
    L.Loop: ([], [L.Any]),
    L.EndLoop: ([L.Loop], [L.Any]),
    L.Subtask: ([], [L.Any]),
    L.Padding: ([], []),
}


def classify(
    actions: np.ndarray, instruction: np.ndarray, successes: np.ndarray
) -> float:
    import ipdb  # type:ignore

    ipdb.set_trace()
    raise NotImplementedError


def main(root: Path, path: Path) -> None:
    path = Path(root, path)
    actions = np.load(Path(path, "eval_actions.npz"))
    instructions = np.load(Path(path, "eval_instruction.npz"))
    successes = np.load(Path(path, "eval_successes.npy"))
    assert len(actions) == len(instructions) == len(successes)

    def iterator() -> Generator[float, None, None]:
        for args in tqdm(
            zip(actions.values(), instructions.values(), successes), total=len(actions)
        ):
            yield classify(*args)

    print(list(iterator()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    main(**vars(parser.parse_args()))
