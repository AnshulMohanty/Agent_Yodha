"""Let the teacher play generated maps alone and report per-tier success (training-only diagnostic).

Usage: uv run python -m training.teacher.run_teacher --maps 10
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from doom_env import make_env
from training.mapgen.grid import generate
from training.mapgen.udmf import write_map
from training.teacher.teacher import Teacher

warnings.filterwarnings("ignore")


def play(tier: int, seed: int, record: str | None = None) -> dict:
    spec = generate(tier, seed)
    env = make_env(str(write_map(spec, f"/tmp/yodha_teacher/t{tier}_{seed}.wad")), "MAP01", episode_ticks=spec.episode_ticks)
    obs, _ = env.reset(seed=seed, options={"record_path": record} if record else None)
    teacher = Teacher(spec, np.random.default_rng(seed))
    steps, info, term, trunc = 0, {}, False, False
    while True:
        st = env.privileged_state()
        a, _ = teacher.act(obs, st)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            break
    st = env.privileged_state()
    monsters = sum(1 for t in spec.things if t.kind == "monster")
    n_keys = len(spec.keys)
    out = dict(tier=tier, exit=bool(info.get("exit_reached")), dead=st["dead"], steps=steps,
               keys=int((st["keys"] > 0).sum()), n_keys=n_keys, kills=st["kills"], monsters=monsters,
               health=st["health"])
    env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()
    for tier in (1, 2, 3, 4):
        t0 = time.time()
        res = [play(tier, a.seed + i) for i in range(a.maps)]
        ex = np.mean([r["exit"] for r in res])
        dead = np.mean([r["dead"] for r in res])
        keys = sum(r["keys"] for r in res) / max(1, sum(r["n_keys"] for r in res))
        kills = sum(r["kills"] for r in res) / max(1, sum(r["monsters"] for r in res))
        print(f"TIER {tier}: exit {ex:.0%}  died {dead:.0%}  keys {keys:.0%}  kills {kills:.0%}  "
              f"avg steps {np.mean([r['steps'] for r in res]):.0f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
