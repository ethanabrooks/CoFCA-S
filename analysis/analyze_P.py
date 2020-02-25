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


def compute_jump(instruction, dest, _from, backward) -> int:
    raise NotImplementedError


def compute_cross_entropy(P: torch.Tensor, instruction: np.ndarray) -> float:
    def compute_with_ptr(ptr: int, done: List[int]) -> float:
        if ptr in done or ptr >= len(instruction):
            return 0
        # print(f"ptr: {ptr}")

        def cross_entropy(jump: int) -> float:
            p = P[ptr].T  # type: ignore
            no_op = P.size(1) // 2
            j = torch.tensor([jump + no_op] * P.size(-1)).cuda()
            return F.cross_entropy(p, j, reduction="none").min().item()

        def cross_entropy_with_dest(dest: L, backward: bool) -> float:
            def compute_jump_to(dest: L) -> Optional[int]:
                if dest == L.Any:
                    assert not backward
                    return 1
                i = torch.tensor(instruction).roll(shifts=-int(ptr), dims=0)
                hits, = np.where(i[:, 0] == dest.value - 1)
                if hits.size:
                    # not empty
                    if backward:
                        return hits[-1] - len(instruction)
                    else:
                        return hits[0]
                return None

            jump = compute_jump_to(dest=dest)
            if jump is None:
                # dest does not exist (e.g. else)
                return 0
            return min(
                (cross_entropy(jump) + compute_with_ptr(ptr + jump, done=done + [ptr]))
                for jump in (jump, jump + 1)
            )

        backward_edges, forward_edges = EDGES[L(instruction[ptr, 0] + 1)]
        backward_cross_entropy = sum(
            cross_entropy_with_dest(dest, backward=True) for dest in backward_edges
        )
        forward_cross_entropy = sum(
            cross_entropy_with_dest(dest, backward=False) for dest in forward_edges
        )
        return backward_cross_entropy + forward_cross_entropy

    return compute_with_ptr(0, done=[])


def main(root: Path, path: Path) -> None:
    path = Path(root, path)
    print("loading P...")
    Ps = torch.load(Path(path, "eval_P.torch"))
    print("loading instructions...")
    instructions = np.load(Path(path, "eval_instruction.npz"))

    def compute_cross_entropies() -> Generator[float, None, None]:
        for args in tqdm(zip(Ps.unbind(dim=1), instructions.values())):
            yield compute_cross_entropy(*args)

    print(list(compute_cross_entropies()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runs/logdir"))
    parser.add_argument("--path", type=Path, required=True)
    main(**vars(parser.parse_args()))
