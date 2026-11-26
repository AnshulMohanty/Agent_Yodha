"""Autopilot: runs on the Mac next to training so nobody has to check by hand. Every few hours it
  1. checks that training is alive (restarts it from latest.pt if it stopped; at most 3 times),
  2. tests the newest model on the fixed held-out maps (same maps every time, so scores are comparable),
  3. records two preview videos: tier 1 (fight arena) and tier 3 (key -> locked door -> exit),
  4. writes runs/STATUS.md in plain language,
  5. if the new model is clearly better than the submitted one: installs it, runs validate.py, rebuilds the
     brief, commits and pushes (nothing is pushed if validation fails),
  6. shows a macOS notification.

Start:  PYTHONPATH=. nohup uv run python -m training.eval.autopilot > runs/autopilot.log 2>&1 &
Stop:   pkill -f training.eval.autopilot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "mac_main"
LOG = ROOT / "runs" / "mac_main.log"
HIST = ROOT / "runs" / "autopilot_history.json"
STATUS = ROOT / "runs" / "STATUS.md"
VIDEOS = ROOT / "runs" / "videos"
ODO = ROOT / "submission" / "weights" / "odometry.json"
TRAIN_CMD = ["caffeinate", "-dimsu", "uv", "run", "python", "-m", "training.algo.kickstart_ppo",
             "--config", "training/configs/mac.yaml", "--set", "init_ckpt=runs/mac_main/latest.pt"]
PREVIEWS = [(1, 900_001_002, "fight arena"), (3, 900_003_001, "key -> locked door -> exit")]
TIER_NAME = {1: "Tier 1 fight arenas", 2: "Tier 2 rooms + doors", 3: "Tier 3 key -> lock -> exit",
             4: "Tier 4 chained locks + ambushes"}
PHASE_NAME = {"A": "A: copying the coach", "B": "B: copying the coach + learning from rewards",
              "C": "C: learning from rewards"}


def training_alive() -> bool:
    return subprocess.run(["pgrep", "-f", "training.algo.kickstart_ppo"], capture_output=True).returncode == 0


def restart_training() -> None:
    log = open(LOG, "a")
    subprocess.Popen(TRAIN_CMD, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "PYTHONPATH": str(ROOT)})


def last_train_line() -> dict:
    if not LOG.exists():
        return {}
    lines = [ln for ln in LOG.read_text(errors="ignore").splitlines()[-400:] if re.match(r"\[\d+\] steps", ln)]
    if not lines:
        return {}
    ln = lines[-1]
    out = {"line": ln}
    m = re.search(r"steps ([\d.]+)M phase (\w)", ln)
    if m:
        out["steps_m"], out["phase"] = float(m.group(1)), m.group(2)
    m = re.search(r"sps (\d+)", ln)
    if m:
        out["sps"] = int(m.group(1))
    return out


def snapshot() -> tuple[Path, float] | None:
    import torch

    src = RUN / "export"
    for _ in range(5):
        try:
            steps = json.loads((src / "config.json").read_text()).get("steps", 0)
            dst = ROOT / "runs" / "snaps" / f"s{steps / 1e6:.2f}M"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            if not (dst / "odometry.json").exists():
                shutil.copy(ODO, dst / "odometry.json")
            torch.load(dst / "yodha.pt", map_location="cpu", weights_only=False)  # not mid-write
            return dst, steps / 1e6
        except Exception as e:  # export being rewritten right now
            print("snapshot retry:", e, flush=True)
            time.sleep(30)
    return None


def submitted_score() -> tuple[float, str]:
    cfg = json.loads((ROOT / "submission" / "weights" / "config.json").read_text())
    if "heldout_proxy_score" in cfg:
        return float(cfg["heldout_proxy_score"]), cfg.get("source", "?")
    f = ROOT / "runs" / "eval" / "agent_0p3M_t07.json"
    return (json.loads(f.read_text())["weighted_score"] if f.exists() else 0.0), cfg.get("source", "?")


def record_previews(snap: Path, tag: str) -> list[str]:
    from training.viz.glassbox import run as glassbox

    out = []
    (VIDEOS / "history").mkdir(parents=True, exist_ok=True)
    for tier, seed, _ in PREVIEWS:
        raw = VIDEOS / f"_raw_T{tier}.mp4"
        try:
            res = glassbox(str(snap), tier, seed, str(raw), 0.7)
        except Exception as e:
            print("preview failed:", e, flush=True)
            continue
        latest = VIDEOS / f"latest_T{tier}.mp4"
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "26", str(latest)], check=False)
            raw.unlink(missing_ok=True)
        else:
            shutil.move(raw, latest)
        shutil.copy(latest, VIDEOS / "history" / f"T{tier}_{tag}.mp4")
        out.append(f"{latest.relative_to(ROOT)} -> exit {'YES' if res['exit'] else 'no'}, "
                   f"{'died' if res['dead'] else 'alive'}, {res['kills']} kills, {res['steps']} decisions")
    return out


def pct(x: float) -> str:
    return f"{x:.0%}"


def write_status(train: dict, alive: bool, s: dict | None, tag: str, sub: tuple[float, str], promoted: str,
                 videos: list[str], hist: list, next_at: datetime) -> None:
    L = [f"# YODHA status (updated {datetime.now():%H:%M}, next check about {next_at:%H:%M})", ""]
    if alive:
        L.append(f"- **Training:** RUNNING. {train.get('steps_m', 0):.2f}M steps, phase "
                 f"{PHASE_NAME.get(train.get('phase'), '?')}, {train.get('sps', 0)} steps/s")
    else:
        L.append("- **Training:** NOT running (see runs/mac_main.log)")
    L.append(f"- **Submitted model** (on GitHub): held-out score **{sub[0]:.0f}** ({sub[1]})")
    if s is not None:
        L.append(f"- **Newest model** ({tag}): held-out score **{s['weighted_score']:.0f}**. {promoted}")
        L += ["", "## What the newest model can do (6 unseen maps per tier, temperature 0.7)", "",
              "| Level type | Finished (exit) | Died | Keys picked | Locked doors opened | Enemies killed | Area explored |",
              "|---|---|---|---|---|---|---|"]
        for tier, t in s["tiers"].items():
            L.append(f"| {TIER_NAME[int(tier)]} | {pct(t['exit'])} | {pct(t['dead'])} | {pct(t['keys'])} | "
                     f"{pct(t['locks'])} | {pct(t['kills'])} | {pct(t['coverage'])} |")
        L += ["", "For comparison: random button-mashing scores about -32, the coach (privileged teacher) about 640."]
    if videos:
        L += ["", "## Watch it (double-click the file, or `open <file>`)"] + [f"- {v}" for v in videos]
        L.append("- Older videos: runs/videos/history/. Make a new one any time: `scripts/preview.sh 3` (tier 1-4)")
    L += ["", "## History", "", "| time | steps | held-out score | submitted? |", "|---|---|---|---|"]
    for h in hist[-20:]:
        L.append(f"| {h['time']} | {h['steps_m']:.2f}M | {h['score']:.0f} | {'YES' if h['promoted'] else ''} |")
    STATUS.write_text("\n".join(L) + "\n")


def notify(msg: str) -> None:
    subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "YODHA autopilot"'],
                   capture_output=True)


def cycle(margin: float, push: bool, maps: int, restarts: list, interval_h: float) -> None:
    from training.eval.harness import evaluate, print_summary
    from training.eval.promote import install

    print(f"\n=== autopilot cycle {datetime.now():%H:%M}", flush=True)
    alive = training_alive()
    if not alive and len(restarts) < 3:
        print("training not running -> restarting from latest.pt", flush=True)
        restart_training()
        restarts.append(time.time())
        notify("Training had stopped - restarted from the last checkpoint")
        time.sleep(90)
        alive = training_alive()
    train = last_train_line()
    hist = json.loads(HIST.read_text()) if HIST.exists() else []
    sub = submitted_score()
    snap = snapshot()
    s, promoted, videos, tag = None, "", [], "?"
    if snap is not None:
        snap_dir, steps_m = snap
        tag = f"{steps_m:.2f}M"
        if hist and abs(hist[-1]["steps_m"] - steps_m) < 1e-6:
            print("no new export since last check", flush=True)
        else:
            s = evaluate(str(snap_dir), "agent", maps, 2, 0.7)
            print_summary(s)
            (ROOT / "runs" / "eval" / f"auto_{tag}.json").write_text(json.dumps(s, indent=1, default=float))
            ok = False
            if s["weighted_score"] > sub[0] + margin:
                ok = install(s, snap_dir, 0.7, push)
                promoted = "Better than the submitted one, so it was **installed, validated and pushed**." if ok \
                    else "Better, but validate.py failed, so the old submission was kept."
                if ok:
                    sub = submitted_score()
            else:
                promoted = "Not clearly better than the submitted one, so the submission was kept."
            hist.append({"time": f"{datetime.now():%d %b %H:%M}", "steps_m": steps_m,
                         "score": s["weighted_score"], "promoted": ok})
            HIST.write_text(json.dumps(hist, indent=1))
            videos = record_previews(snap_dir, tag)
            notify(f"{tag} steps: score {s['weighted_score']:.0f} (submitted {sub[0]:.0f})"
                   + (" - NEW SUBMISSION PUSHED" if ok else ""))
    write_status(train, alive, s, tag, sub, promoted, videos, hist, datetime.now() + timedelta(hours=interval_h))
    print(STATUS.read_text(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0, help="time between checks")
    ap.add_argument("--margin", type=float, default=15.0, help="promote only if better by this much")
    ap.add_argument("--maps", type=int, default=6)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    restarts: list = []
    while True:
        try:
            cycle(a.margin, not a.no_push, a.maps, restarts, a.hours)
        except Exception as e:  # never let the autopilot die
            print("cycle error:", repr(e), flush=True)
        if a.once:
            break
        time.sleep(a.hours * 3600)


if __name__ == "__main__":
    main()
