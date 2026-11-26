"""Pick the best checkpoint (and sampling temperature) on held-out maps and install it as the submission.

Usage:
  uv run python -m training.eval.promote --candidates runs/mac_main/export runs/kaggle_b_export --maps 6 [--push]
Each candidate is an export dir (yodha.pt + config.json). The winner's weights go to submission/weights/,
the brief's results file runs/eval/agent.json is updated, validate.py is run, and with --push the
result is committed and pushed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from training.eval.harness import evaluate, print_summary

ROOT = Path(__file__).resolve().parents[2]
ODO = ROOT / "submission" / "weights" / "odometry.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.5, 0.7, 1.0])
    ap.add_argument("--maps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    best = None
    for cand in a.candidates:
        c = Path(cand)
        if not (c / "yodha.pt").exists():
            print("skip (no yodha.pt):", c)
            continue
        if not (c / "odometry.json").exists():
            shutil.copy(ODO, c / "odometry.json")
        for t in a.temps:
            s = evaluate(str(c), "agent", a.maps, a.workers, t)
            print_summary(s)
            if best is None or s["weighted_score"] > best[0]["weighted_score"]:
                best = (s, c, t)
    if best is None:
        raise SystemExit("no valid candidates")
    s, c, t = best
    print(f"\nBEST: {c} temperature {t} score {s['weighted_score']:.1f}")
    dst = ROOT / "submission" / "weights"
    shutil.copy(c / "yodha.pt", dst / "yodha.pt")
    cfg = json.loads((c / "config.json").read_text()) if (c / "config.json").exists() else {}
    cfg.update({"temperature": t, "source": str(c), "heldout_proxy_score": round(s["weighted_score"], 1)})
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    (ROOT / "runs" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "eval" / "agent.json").write_text(json.dumps(s, indent=1, default=float))
    v = subprocess.run(["uv", "run", "python", "validate.py"], cwd=ROOT, capture_output=True, text=True)
    print(v.stdout[-800:])
    if "PASSED" not in v.stdout:
        raise SystemExit("validate.py did not pass: NOT pushing")
    subprocess.run(["uv", "run", "python", "brief/make_brief.py"], cwd=ROOT, check=False)
    if a.push:
        subprocess.run(["git", "add", "submission/weights", "brief"], cwd=ROOT, check=True)
        msg = (f"Promote {c.name} (T={t}): held-out proxy score {s['weighted_score']:.0f}\n\n"
               "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("pushed")


if __name__ == "__main__":
    main()
