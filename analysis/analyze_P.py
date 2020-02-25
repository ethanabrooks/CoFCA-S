#! /usr/bin/env python
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum
from typing import Dict, Tuple, List, Generator

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


def compute_jump(instruction, dest, _from, backward) -> int:
    raise NotImplementedError


def compute_cross_entropy(P: torch.Tensor, instruction: np.ndarray) -> float:
    cache: Dict[int, float] = {}

    def compute_with_ptr(ptr):
        if ptr in cache:
            return cache[ptr]
        def cross_entropy(jump: int) -> float:
            p = P[ptr].T  # type: ignore
            no_op = P.size(1) // 2
            j = torch.tensor([jump + no_op] * P.size(-1)).cuda()
            return F.cross_entropy(p, j, reduction="none").min().item()

        def cross_entropy_with_dest(dest: L, backward: bool) -> float:
            def compute_jump_to(dest: L) -> int:
                if dest == L.Any:
                    assert not backward
                    return 1
                i = torch.tensor(instruction).roll(shifts=-int(ptr), dims=0)
                hits, = np.where(i[:, 0] == dest.value - 1)
                if backward:
                    return hits[-1] - len(instruction)
                else:
                    return hits[0]

            def recurse(jump):
                k = ptr + jump
                import ipdb

                ipdb.set_trace()
                if k in cache:
                    return cache[k]
                cross_entropy = compute_with_ptr(k)
                cache[k] = cross_entropy
                return cross_entropy

            jump = compute_jump_to(dest=dest)
            return min(
                (cross_entropy(jump) + recurse(jump)) for jump in (jump, jump + 1)
            )

        backward_edges, forward_edges = EDGES[L(instruction[ptr, 0] + 1)]
        backward_cross_entropy = sum(
            cross_entropy_with_dest(dest, backward=True) for dest in backward_edges
        )
        forward_cross_entropy = sum(
            cross_entropy_with_dest(dest, backward=False) for dest in forward_edges
        )
        return cross_entropy

    return sum(map(compute_with_ptr, range(len(instructions))


def main(root: Path, path: Path) -> None:
    path = Path(root, path)
    print("loading P...")
    Ps = torch.load(Path(path, "eval_P.torch"))
    print("loading instructions...")
    instructions = np.load(Path(path, "eval_instruction.npz"))

    def compute_cross_entropies() -> Generator[float, None, None]:
        for args in zip(Ps.unbind(dim=1), instructions.values()):
            yield compute_cross_entropy(*args)

    print(list(compute_cross_entropies()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    main(**vars(parser.parse_args()))
