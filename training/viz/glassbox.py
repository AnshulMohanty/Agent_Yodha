"""Glass-box video: game view + EgoMap (fine/coarse) + intent and action probabilities, per decision.

Usage: uv run python -m training.viz.glassbox --weights runs/mac_main/export --tier 3 --seed 900003001 --out runs/videos/t3.mp4
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np

from doom_env import make_env
from submission.yodha.model import HEADS, INTENTS
from submission.yodha.policy import YodhaPolicy
from training.mapgen.grid import generate
from training.mapgen.udmf import write_map

warnings.filterwarnings("ignore")
# EgoMap channel colours (BGR): free, wall, door, lock, enemy, item, key, visited, projectile, frontier
COLORS = [(60, 60, 60), (200, 200, 200), (0, 200, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 255),
          (120, 80, 40), (0, 128, 255), (255, 255, 0)]
DRAW_ORDER = [0, 7, 9, 1, 2, 3, 5, 6, 4, 8]


def render_map(crop: np.ndarray, size: int) -> np.ndarray:
    img = np.zeros(crop.shape[1:] + (3,), np.uint8)
    for ch in DRAW_ORDER:
        m = crop[ch] > 40
        img[m] = COLORS[ch]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    c = size // 2
    cv2.drawMarker(img, (c, c), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    return img


def bars(probs: np.ndarray, labels, w: int, title: str) -> np.ndarray:
    h = 22 * len(labels) + 26
    img = np.full((h, w, 3), 25, np.uint8)
    cv2.putText(img, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, (p, lab) in enumerate(zip(probs, labels)):
        y = 26 + 22 * i
        cv2.rectangle(img, (110, y + 3), (110 + int((w - 120) * float(p)), y + 17), (80, 200, 80), -1)
        cv2.putText(img, f"{lab}", (6, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return img


def run(weights: str, tier: int, seed: int, out: str, temperature: float = 0.0, max_steps: int = 2000) -> dict:
    spec = generate(tier, seed)
    env = make_env(str(write_map(spec, f"/tmp/yodha_viz/t{tier}_{seed}.wad")), "MAP01", episode_ticks=spec.episode_ticks)
    obs, _ = env.reset(seed=seed % 100_000)
    pol = YodhaPolicy(weights)
    pol.temperature = temperature
    pol.reset()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    W, H = 1280, 720
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 9, (W, H))
    move_labels = ["-", "fwd", "back"]
    info = {}
    for step in range(max_steps):
        a = pol.act(obs)
        last = pol.last
        frame = np.zeros((H, W, 3), np.uint8)
        game = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
        frame[0:480, 0:640] = cv2.resize(game, (640, 480), interpolation=cv2.INTER_NEAREST)
        frame[0:320, 650:970] = render_map(last["inp"]["map_fine"], 320)
        frame[0:300, 975:1275] = render_map(last["inp"]["map_coarse"], 300)
        ib = bars(last["intent"], INTENTS, 310, "INTENT (what YODHA is doing)")
        frame[335:335 + ib.shape[0], 650:960] = ib
        head_txt = []
        for (name, n), p in zip(HEADS, last["heads"]):
            labels = {"move": move_labels, "strafe": ["-", "left", "right"], "turn": ["-", "left", "right"]}.get(name, ["no", "yes"])
            head_txt.append(f"{name}: {labels[int(np.argmax(p))]} ({p.max():.0%})")
        for i, t in enumerate(head_txt):
            cv2.putText(frame, t, (975, 360 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        v = obs["vitals"]
        keys = "".join(k for k, f in zip("BYRbyr", v[8:14]) if f > 0)
        cv2.putText(frame, f"tier {tier}  step {step}  health {v[0]:.0f}  armor {v[1]:.0f}  ammo {v[3]:.0f}  keys [{keys}]",
                    (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "fine map (+-1024u)        coarse map (+-2048u)", (650, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, "map: grey=free white=wall orange=door red=locked/enemy green=item magenta=key cyan=frontier brown=visited",
                    (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        vw.write(frame)
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    st = env.privileged_state()
    vw.release()
    env.close()
    res = {"exit": bool(info.get("exit_reached")), "dead": st["dead"], "kills": st["kills"], "steps": step + 1}
    print(out, res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tier", type=int, default=3)
    ap.add_argument("--seed", type=int, default=900_003_001)
    ap.add_argument("--out", default="runs/videos/demo.mp4")
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    run(a.weights, a.tier, a.seed, a.out, a.temperature)


if __name__ == "__main__":
    main()
