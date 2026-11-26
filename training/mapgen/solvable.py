"""Solvability check: BFS over (cell, keys held). Training-only."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .grid import DIRS, DOOR, EXIT, FLOOR, LOCKS, SWITCH, MapSpec

COLOR_OF_LOCK = {"R": "red", "B": "blue", "Y": "yellow"}
COLOR_BIT = {"red": 1, "blue": 2, "yellow": 4}


@dataclass
class Solvability:
    solvable: bool
    steps: int  # shortest number of cell moves to finish (with key detours), -1 if unsolvable


def check(spec: MapSpec) -> Solvability:
    grid = spec.grid
    H, W = grid.shape
    key_at = {cell: COLOR_BIT[color] for color, cell in spec.keys}
    start_keys = key_at.get(spec.start, 0)
    seen = {(spec.start, start_keys)}
    q = deque([(spec.start, start_keys, 0)])
    while q:
        (r, c), keys, d = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            ch = grid[nr, nc]
            if ch == EXIT or ch == SWITCH:
                return Solvability(True, d + 1)
            if ch in LOCKS and not keys & COLOR_BIT[COLOR_OF_LOCK[ch]]:
                continue
            if ch not in (FLOOR, DOOR) and ch not in LOCKS:
                continue
            nk = keys | key_at.get((nr, nc), 0)
            state = ((nr, nc), nk)
            if state not in seen:
                seen.add(state)
                q.append(((nr, nc), nk, d + 1))
    return Solvability(False, -1)
