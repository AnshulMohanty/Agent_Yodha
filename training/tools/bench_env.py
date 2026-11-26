"""Measure doom_env speed: env init time and steps/s with and without pixel buffers.

Usage: uv run python training/tools/bench_env.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import vizdoom

from doom_env import make_env

FREEDOOM2 = os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad")
CASES = [("maps/two_rooms.wad", "MAP01"), (FREEDOOM2, "MAP01"), (FREEDOOM2, "MAP03")]


def bench(wad: str, mp: str, render: bool, steps: int = 600) -> tuple[float, float]:
    t0 = time.perf_counter()
    env = make_env(wad, mp, episode_ticks=4200, render_buffers=render)
    env.reset(seed=0)
    init_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    n, t0 = 0, time.perf_counter()
    while n < steps:
        a = (rng.random(9) < 0.3).astype(np.int8)
        a[0] = 1  # mostly walk forward so the player moves around
        _, _, term, trunc, _ = env.step(a)
        n += 1
        if term or trunc:
            env.reset()
    sps = n / (time.perf_counter() - t0)
    env.close()
    return init_s, sps


if __name__ == "__main__":
    for wad, mp in CASES:
        for render in (False, True):
            init_s, sps = bench(wad, mp, render)
            print(f"{os.path.basename(wad):16s} {mp}  render={str(render):5s}  init={init_s:5.2f}s  steps/s={sps:7.0f}")
