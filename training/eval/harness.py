"""Scorer replica: play held-out maps through the real DoomAgent/YodhaPolicy and compute a proxy score.

Held-out maps use seeds that training never draws (fixed list per tier). The proxy mirrors the
official categories: exit (+health/time bonus), keys, doors, locked doors, sector coverage, kills,
damage and early death; tiers are weighted 1/2/3/4.

Usage:
  uv run python -m training.eval.harness --weights runs/mac_main/export --maps 8
  uv run python -m training.eval.harness --random --maps 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HELDOUT_SEED = 900_000_000  # training draws seeds uniformly from [0, 2^31); these are fixed and logged
TIER_WEIGHT = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}


def proxy_score(m: dict) -> float:
    s = 0.0
    if m["exit"]:
        s += 50 + 5 * max(m["health"], 0) / 100 + 5 * (1 - m["tic"] / m["ticks"])
    s += 10 * m["keys"] + 3 * m["doors"] + 8 * m["locks"] + 20 * m["coverage"]
    s += 2 * m["kills"] - 0.02 * m["damage"]
    if m["dead"] and m["tic"] < 0.5 * m["ticks"]:
        s -= 10
    return s


def play_one(args) -> dict:
    tier, idx, weights, mode, temperature = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from doom_env import make_env
    from training.mapgen.grid import generate
    from training.mapgen.udmf import _sectors, write_map

    seed = HELDOUT_SEED + tier * 1000 + idx
    spec = generate(tier, seed)
    path = write_map(spec, f"/tmp/yodha_eval/{os.getpid()}_t{tier}_{idx}.wad")
    env = make_env(str(path), "MAP01", episode_ticks=spec.episode_ticks)
    obs, _ = env.reset(seed=seed % 100_000)
    sid, _ = _sectors(spec.grid)
    if mode == "teacher":
        from training.teacher.teacher import Teacher

        agent = Teacher(spec, np.random.default_rng(seed))
    elif mode == "random":
        agent = None
        rng = np.random.default_rng(seed)
    else:
        from submission.yodha.policy import YodhaPolicy

        agent = YodhaPolicy(weights, device="cpu")
        agent.temperature = temperature
        agent.reset()
    sectors, doors, act_ms = set(), set(), []
    info, term, trunc = {}, False, False
    while True:
        st = env.privileged_state()
        r, c = int(st["y"] // 64), int(st["x"] // 64)
        if 0 <= r < sid.shape[0] and 0 <= c < sid.shape[1] and sid[r, c] >= 0:
            sectors.add(int(sid[r, c]))
        t0 = time.perf_counter()
        if mode == "teacher":
            a, _ = agent.act(obs, st)
        elif mode == "random":
            a = (rng.random(9) < 0.3).astype(np.int8)
        else:
            a = agent.act(obs)
        act_ms.append((time.perf_counter() - t0) * 1000)
        obs, _, term, trunc, info = env.step(a)
        ps = env.privileged_state()
        for sec, lock in env.mapinfo.door_sectors.items():
            if ps["sector_ceiling"][sec] - ps["sector_floor"][sec] > 56:
                doors.add((sec, lock))
        if term or trunc:
            break
    st = env.privileged_state()
    env.close()
    m = dict(tier=tier, idx=idx, exit=bool(info.get("exit_reached")), dead=bool(st["dead"]), tic=st["tic"],
             ticks=spec.episode_ticks, health=st["health"], keys=int((st["keys"] > 0).sum()), n_keys=len(spec.keys),
             doors=sum(1 for _, lk in doors if lk == 0), locks=sum(1 for _, lk in doors if lk > 0), n_locks=len(spec.locks),
             coverage=len(sectors) / max(1, int(sid.max()) + 1), kills=st["kills"],
             monsters=sum(1 for t in spec.things if t.kind == "monster"), damage=st["damage_taken"],
             act_ms_p50=float(np.median(act_ms)), act_ms_p99=float(np.percentile(act_ms, 99)))
    m["score"] = proxy_score(m)
    return m


def evaluate(weights: str | None, mode: str, maps: int, workers: int, temperature: float = 0.0) -> dict:
    jobs = [(tier, i, weights, mode, temperature) for tier in (1, 2, 3, 4) for i in range(maps)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(play_one, jobs))
    summary = {"mode": mode, "weights": weights, "temperature": temperature, "tiers": {}}
    total = 0.0
    for tier in (1, 2, 3, 4):
        rs = [r for r in results if r["tier"] == tier]
        t = dict(
            exit=float(np.mean([r["exit"] for r in rs])), dead=float(np.mean([r["dead"] for r in rs])),
            keys=sum(r["keys"] for r in rs) / max(1, sum(r["n_keys"] for r in rs)),
            locks=sum(r["locks"] for r in rs) / max(1, sum(r["n_locks"] for r in rs)),
            coverage=float(np.mean([r["coverage"] for r in rs])),
            kills=sum(r["kills"] for r in rs) / max(1, sum(r["monsters"] for r in rs)),
            score=float(np.mean([r["score"] for r in rs])),
            act_ms_p99=float(np.max([r["act_ms_p99"] for r in rs])),
        )
        summary["tiers"][tier] = t
        total += TIER_WEIGHT[tier] * t["score"]
    summary["weighted_score"] = total
    summary["maps"] = results
    return summary


def print_summary(s: dict) -> None:
    print(f"== {s['mode']} {s['weights'] or ''} T={s['temperature']}  WEIGHTED PROXY SCORE {s['weighted_score']:.1f}")
    for tier, t in s["tiers"].items():
        print(f"  T{tier}: score {t['score']:6.1f} | exit {t['exit']:.0%} dead {t['dead']:.0%} keys {t['keys']:.0%} "
              f"locks {t['locks']:.0%} cov {t['coverage']:.0%} kills {t['kills']:.0%} | act p99 {t['act_ms_p99']:.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="export dir with yodha.pt/config.json/odometry.json")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--maps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mode = "teacher" if a.teacher else "random" if a.random else "agent"
    weights = a.weights
    if mode == "agent" and weights:
        w = Path(weights)
        odo = Path(__file__).resolve().parents[2] / "submission" / "weights" / "odometry.json"
        if not (w / "odometry.json").exists() and odo.exists():
            (w / "odometry.json").write_text(odo.read_text())
    s = evaluate(weights, mode, a.maps, a.workers, a.temperature)
    print_summary(s)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
