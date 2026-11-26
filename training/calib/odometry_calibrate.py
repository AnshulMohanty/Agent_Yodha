"""Fit the agent's odometry model from gameplay (training-only; privileged pose is the ground truth).

Fits the heading table (keyed on recent turn-button history, with special keys for the first two
steps of a map) and the translation matrix used by submission/yodha/odometry.py, then measures the
drift of the *actual agent code* (Odometry class) against privileged pose. Saves
submission/weights/odometry.json.

Usage: uv run python -m training.calib.odometry_calibrate --episodes 60
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from doom_env import make_env
from submission.yodha.odometry import Odometry, heading_key, turn_of
from training.mapgen.grid import generate
from training.mapgen.udmf import write_map

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "submission" / "weights" / "odometry.json"


def random_policy(rng):
    """Persistent random actions (held for 1-12 steps), roughly like a playing agent."""
    while True:
        a = np.zeros(9, np.int8)
        a[0] = rng.random() < 0.6
        a[1] = (not a[0]) and rng.random() < 0.15
        s = rng.random()
        a[2], a[3] = s < 0.12, 0.12 <= s < 0.24
        t = rng.random()
        a[4], a[5] = t < 0.25, 0.25 <= t < 0.5
        a[6] = rng.random() < 0.2
        for _ in range(int(rng.integers(1, 13))):
            yield a.copy()


def collect(episodes: int, steps: int, seed: int):
    """Episodes as lists of (action, vitals, true_x, true_y, true_angle), plus the true start pose."""
    rng = np.random.default_rng(seed)
    eps = []
    for ep in range(episodes):
        spec = generate(int(rng.integers(1, 5)), seed * 100_000 + ep)
        env = make_env(str(write_map(spec, f"/tmp/yodha_calib/{seed}_{ep}.wad")), "MAP01", episode_ticks=4 * steps + 8)
        env.reset(seed=ep)
        st = env.privileged_state()
        start = (st["x"], st["y"], st["angle"])
        pol, seq = random_policy(rng), []
        for _ in range(steps):
            a = next(pol)
            obs, _, term, trunc, _ = env.step(a)
            st = env.privileged_state()
            seq.append((a, obs["vitals"].copy(), st["x"], st["y"], st["angle"]))
            if term or trunc:
                break
        env.close()
        eps.append((start, seq))
    return eps


def to_ego(start, seq):
    """True pose relative to the start pose (agent frame: start at origin, heading 0)."""
    x0, y0, a0 = start
    c, s = math.cos(math.radians(-a0)), math.sin(math.radians(-a0))
    out = []
    for _, _, x, y, a in seq:
        dx, dy = x - x0, y - y0
        out.append((dx * c - dy * s, dx * s + dy * c, (a - a0 + 180) % 360 - 180))
    return out


def fit(eps) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    X, Y = [], []
    for start, seq in eps:
        prev_angle, hist, prev_v, prev_pos = start[2], [0, 0, 0, 0], np.zeros(2), start[:2]
        for k, (a, vit, x, y, ang) in enumerate(seq):
            hist = [turn_of(a)] + hist[:3]
            dth = (ang - prev_angle + 180) % 360 - 180
            for depth in (4, 3):
                groups[heading_key(k, hist, depth)].append(dth)
            th, tr = math.radians(ang), math.radians(dth)
            dx, dy = x - prev_pos[0], y - prev_pos[1]
            Y.append((dx * math.cos(th) + dy * math.sin(th), dx * math.sin(th) - dy * math.cos(th)))
            pf, pl = prev_v[0], -prev_v[1]
            X.append((vit[14], vit[15], pf * math.cos(tr) + pl * math.sin(tr), -(-pf * math.sin(tr) + pl * math.cos(tr))))
            prev_angle, prev_v, prev_pos = ang, np.array(vit[14:16], np.float64), (x, y)
    table = {k: float(np.median(v)) for k, v in groups.items() if len(v) >= 3}
    A, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
    return {"turn_table": table, "trans_A": A.tolist()}


def evaluate(eps, calib: dict) -> dict:
    odo = Odometry(calib)
    pos_err, head_err, paths, per100 = [], [], [], []
    for start, seq in eps:
        odo.reset()
        truth = to_ego(start, seq)
        path, last = 0.0, (0.0, 0.0)
        for (a, vit, *_), (tx, ty, ta) in zip(seq, truth):
            x, y, h = odo.update(a, vit)
            path += math.hypot(tx - last[0], ty - last[1])
            last = (tx, ty)
        pos_err.append(math.hypot(x - tx, y - ty))
        head_err.append(abs((h - ta + 180) % 360 - 180))
        paths.append(path)
        per100.append(pos_err[-1] / max(path, 1.0) * 100)
    return {"episodes": len(eps), "final_pos_err_mean": float(np.mean(pos_err)),
            "final_heading_err_mean_deg": float(np.mean(head_err)), "path_len_mean": float(np.mean(paths)),
            "pos_err_pct_of_path_mean": float(np.mean(per100)), "pos_err_pct_of_path_p90": float(np.percentile(per100, 90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()
    train = collect(a.episodes, a.steps, seed=1)
    test = collect(max(10, a.episodes // 3), a.steps, seed=2)
    calib = fit(train)
    calib["drift_heldout"] = evaluate(test, calib)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(calib, indent=1))
    print(json.dumps(calib["drift_heldout"], indent=1))
    print("trans_A:", np.round(np.array(calib["trans_A"]), 3).tolist(), "| table keys:", len(calib["turn_table"]))


if __name__ == "__main__":
    main()
