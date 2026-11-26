"""Final model selection, built to run unattended before the deadline.

  1. waits until --at HH:MM, then stops the autopilot and stops training right after a checkpoint save
  2. candidates: Mac checkpoints from the last --last-m million steps, two weight-averaged models (last 3 / last 5
     checkpoints), autopilot snapshots, the current submission, and every export dir given with --extra
     (e.g. runs/kaggle_b_export) or found at runs/kaggle_b_export
  3. round 1 (screening): every candidate on the 24 usual held-out maps (6 per tier), temperature 0.7
  4. round 2 (decision): the top 3 plus the current submission on 60 FRESH held-out maps (15 per tier, never used
     for any earlier choice) at temperatures 0.5 / 0.7 / 1.0; the best weighted score wins
  5. random + coach baselines on the same 60 maps (for the brief), install the winner (validate.py must pass),
     rebuild the brief, commit and push
  6. demo videos (tier 1, 3, 4), a real Freedoom 2 check, runs/FINAL_REPORT.md and a notification

Run:      caffeinate -dimsu uv run python -m training.eval.finalize --at 06:45 > runs/finalize.log 2>&1 &
Kaggle later (only compares the new dirs with the submitted model on the same 60 maps):
          uv run python -m training.eval.finalize --now --quick runs/kaggle_b_export
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import torch

from training.eval.harness import evaluate, evaluate_real, print_summary
from training.eval.promote import install

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "mac_main"
LOG = ROOT / "runs" / "mac_main.log"
OUT = ROOT / "runs" / "final"
EVAL = ROOT / "runs" / "eval"
SUB = ROOT / "submission" / "weights"
ODO = SUB / "odometry.json"
R2_START, R2_MAPS = 6, 15  # round 2 uses map indices 6..20: disjoint from the 0..5 maps used by the autopilot
REAL_LEVELS = ["MAP01", "MAP02", "MAP03", "MAP04", "MAP05", "MAP06"]


def log(*a) -> None:
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def notify(msg: str) -> None:
    subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "YODHA finalize"'], capture_output=True)


def wait_until(hhmm: str) -> None:
    h, m = map(int, hhmm.split(":"))
    now = datetime.now()
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t < now - timedelta(hours=1):
        t += timedelta(days=1)
    log(f"waiting until {t:%d %b %H:%M}")
    while datetime.now() < t:
        time.sleep(30)


def stop_everything() -> None:
    subprocess.run(["pkill", "-f", "training.eval.autopilot"], capture_output=True)
    if subprocess.run(["pgrep", "-f", "training.algo.kickstart_ppo"], capture_output=True).returncode == 0:
        n0 = LOG.read_text(errors="ignore").count("saved checkpoint")
        t0 = time.time()
        while time.time() - t0 < 240 and LOG.read_text(errors="ignore").count("saved checkpoint") == n0:
            time.sleep(3)
        subprocess.run(["pkill", "-TERM", "-f", "python -m training.algo.kickstart_ppo"], capture_output=True)
        time.sleep(10)
    log("training and autopilot stopped")


def export(ckpts: list[Path], out: Path) -> Path:
    """Inference export of one checkpoint, or the element-wise average of several (same run, nearby steps)."""
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in ckpts]
    sd = {}
    for k, v in cks[-1]["model"].items():
        if k.startswith("value.") or k.startswith("aux."):
            continue
        sd[k] = sum(c["model"][k].float() for c in cks) / len(cks) if v.is_floating_point() else v
    out.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out / "yodha.pt")
    cfg = cks[-1]["cfg"]
    (out / "config.json").write_text(json.dumps({"hidden": cfg["hidden"], "use_pixels": cfg["use_pixels"],
                                                 "temperature": 0.0, "steps": int(cks[-1]["steps"]),
                                                 "averaged": [p.name for p in ckpts] if len(ckpts) > 1 else None}))
    shutil.copy(ODO, out / "odometry.json")
    return out


def copy_dir(src: Path, name: str) -> Path | None:
    if not (src / "yodha.pt").exists():
        log("skip (no yodha.pt):", src)
        return None
    dst = OUT / "cand" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    if not (dst / "odometry.json").exists():
        shutil.copy(ODO, dst / "odometry.json")
    if not (dst / "config.json").exists():
        (dst / "config.json").write_text(json.dumps({"hidden": 512, "use_pixels": True}))
    return dst


def candidates(last_m: int, extra: list[str]) -> list[Path]:
    c = []
    sub = copy_dir(SUB, "submitted")
    c.append(sub)
    nums = sorted(RUN.glob("ckpt_*M.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    if nums:
        top = int(re.findall(r"\d+", nums[-1].name)[0])
        recent = [p for p in nums if int(re.findall(r"\d+", p.name)[0]) > top - last_m]
        for p in recent:
            c.append(export([p], OUT / "cand" / f"mac_{p.stem[5:]}"))
        if (RUN / "latest.pt").exists():
            c.append(export([RUN / "latest.pt"], OUT / "cand" / "mac_latest"))
        for k in (3, 5):
            if len(recent) >= k:
                c.append(export(recent[-k:], OUT / "cand" / f"mac_avg{k}"))
    for d in sorted((ROOT / "runs" / "snaps").glob("s*M")):
        try:
            if float(d.name[1:-1]) >= 9.0:
                c.append(copy_dir(d, f"snap_{d.name[1:]}"))
        except ValueError:
            pass
    kag = [ROOT / "runs" / "kaggle_b_export"] + [Path(e) for e in extra]
    for i, k in enumerate(dict.fromkeys(p.resolve() for p in kag)):
        if k.exists():
            c.append(copy_dir(k, f"kaggle_{i}" if "kaggle" in str(k) else f"extra_{i}"))
            if (k / "latest.pt").exists():
                c.append(export([k / "latest.pt"], OUT / "cand" / f"kaggle_{i}_latest"))
    out, seen = [], set()
    for x in c:  # drop missing and byte-identical candidates (e.g. the submission is also an autopilot snapshot)
        if x is None:
            continue
        h = hashlib.sha1((x / "yodha.pt").read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(x)
    return out


def round2(cands: list[Path], temps: list[float], workers: int, cache: dict) -> list[tuple]:
    res = []
    for c in cands:
        for t in temps:
            key = f"{hashlib.sha1((c / 'yodha.pt').read_bytes()).hexdigest()[:12]}@{t}"  # same weights -> reuse
            if key not in cache:
                try:
                    s = evaluate(str(c), "agent", R2_MAPS, workers, t, start=R2_START)
                except Exception as e:
                    log("round 2 failed for", key, repr(e))
                    continue
                print_summary(s)
                cache[key] = s
                (OUT / "round2.json").write_text(json.dumps(cache, indent=1, default=float))
            res.append((cache[key], c, t))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="06:45")
    ap.add_argument("--now", action="store_true")
    ap.add_argument("--last-m", type=int, default=6)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--quick", nargs="*", default=None, help="only compare these export dirs with the submission")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry", action="store_true", help="tiny test run: does not stop training or touch the submission")
    a = ap.parse_args()
    global OUT, EVAL, R2_MAPS
    r1_maps = 6
    if a.dry:
        OUT, EVAL, R2_MAPS, r1_maps, a.now = OUT / "dry", OUT / "dry" / "eval", 1, 1, True
        EVAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads((OUT / "round2.json").read_text()) if (OUT / "round2.json").exists() else {}
    temps = [0.7] if a.dry else [0.5, 0.7, 1.0]
    rep = [f"# YODHA final selection ({datetime.now():%d %b %H:%M})", ""]

    if a.quick is not None:  # e.g. the Kaggle model arrives after the main finalize
        cands = [copy_dir(SUB, "submitted_now")]
        for i, q in enumerate(a.quick):
            cands.append(copy_dir(Path(q), f"quick_{i}"))
            if (Path(q) / "latest.pt").exists():
                cands.append(export([Path(q) / "latest.pt"], OUT / "cand" / f"quick_{i}_latest"))
        res = round2([c for c in cands if c is not None], temps, a.workers, cache)
        cur = max(r[0]["weighted_score"] for r in res if r[1].name == "submitted_now")
        s, c, t = max(res, key=lambda r: r[0]["weighted_score"])
        log(f"best: {c.name} T={t} score {s['weighted_score']:.1f} (submitted model best {cur:.1f})")
        if a.dry:
            return
        if c.name != "submitted_now" and s["weighted_score"] > cur:
            ok = install(s, c, t, not a.no_push)
            notify(f"New model won ({s['weighted_score']:.0f} vs {cur:.0f})" + (" and was pushed" if ok else ", validate FAILED"))
        else:
            notify(f"Submitted model kept ({cur:.0f} vs best new {s['weighted_score']:.0f})")
        return

    if not a.now:
        wait_until(a.at)
    if not a.dry:
        stop_everything()
        notify("Final selection started (about 40 min)")
    cands = candidates(a.last_m, a.extra)
    log("candidates:", [c.name for c in cands])

    # round 1: screening on the usual 24 maps
    r1 = []
    for c in cands:
        try:
            s = evaluate(str(c), "agent", r1_maps, a.workers, 0.7)
        except Exception as e:
            log("round 1 failed for", c.name, repr(e))
            continue
        print_summary(s)
        r1.append((s["weighted_score"], c))
    r1.sort(key=lambda x: -x[0])
    (OUT / "round1.json").write_text(json.dumps([(sc, c.name) for sc, c in r1], indent=1))
    finalists = [c for _, c in r1[:3]]
    if not any(c.name == "submitted" for c in finalists):
        finalists.append(next(c for c in cands if c.name == "submitted"))
    log("finalists:", [c.name for c in finalists])

    # round 2: fresh 60 maps x 3 temperatures
    res = round2(finalists, temps, a.workers, cache)
    res.sort(key=lambda r: (-r[0]["weighted_score"], sum(m["dead"] for m in r[0]["maps"])))
    best_s, best_c, best_t = res[0]
    log(f"WINNER {best_c.name} T={best_t} score {best_s['weighted_score']:.1f}")

    # baselines on the same 60 maps for the brief
    for mode in ("random", "teacher"):
        p = EVAL / f"{mode}.json"
        if p.exists() and not (EVAL / f"{mode}_24maps.json").exists():
            shutil.copy(p, EVAL / f"{mode}_24maps.json")
        try:
            s = evaluate(None, mode, R2_MAPS, a.workers, 0.0, start=R2_START)
            print_summary(s)
            p.write_text(json.dumps(s, indent=1, default=float))
        except Exception as e:
            log("baseline failed:", mode, repr(e))

    # real Freedoom 2 levels (information for the brief / demo)
    try:
        real = evaluate_real(str(best_c), REAL_LEVELS, min(a.workers, len(REAL_LEVELS)), best_t)
        (EVAL / "real.json").write_text(json.dumps(real, indent=1, default=float))
    except Exception as e:
        real = None
        log("real-level check failed:", repr(e))

    ok = False if a.dry else install(best_s, best_c, best_t, not a.no_push)

    # demo videos of the winner
    from training.viz.glassbox import run as glassbox

    vids = []
    for tier, seed in () if a.dry else ((1, 900_001_002), (3, 900_003_001), (4, 900_004_001)):
        raw, out = OUT / f"_raw_T{tier}.mp4", ROOT / "runs" / "videos" / f"final_T{tier}.mp4"
        try:
            r = glassbox(str(best_c), tier, seed, str(raw), best_t)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt",
                            "yuv420p", "-crf", "26", str(out)], check=False)
            raw.unlink(missing_ok=True)
            vids.append(f"- runs/videos/final_T{tier}.mp4: exit {'YES' if r['exit'] else 'no'}, "
                        f"{'died' if r['dead'] else 'alive'}, {r['kills']} kills, {r['steps']} decisions")
        except Exception as e:
            log("video failed:", tier, repr(e))

    rep += [f"**Winner:** {best_c.name}, temperature {best_t}, score **{best_s['weighted_score']:.0f}** on 60 fresh maps. "
            + ("Installed, validated and pushed." if ok else "NOT installed (validate.py failed); previous submission kept."),
            "", "## Round 1 (24 maps, T=0.7)", "", "| candidate | score |", "|---|---|"]
    rep += [f"| {c.name} | {sc:.0f} |" for sc, c in r1]
    rep += ["", "## Round 2 (60 fresh maps)", "", "| candidate | T | score | T1 exit | T2 exit | T3 exit | T4 exit | deaths |",
            "|---|---|---|---|---|---|---|---|"]
    for s, c, t in res:
        tz = s["tiers"]
        dead = sum(m["dead"] for m in s["maps"])
        rep.append(f"| {c.name} | {t} | {s['weighted_score']:.0f} | " + " | ".join(f"{tz[k]['exit']:.0%}" for k in tz)
                   + f" | {dead}/{len(s['maps'])} |")
    for mode in ("random", "teacher"):
        p = EVAL / f"{mode}.json"
        if p.exists():
            rep.append(f"\nBaseline {mode}: {json.loads(p.read_text())['weighted_score']:.0f} on the same maps")
    if real:
        rep += ["", "## Real Freedoom 2 levels (winner)", "", "| level | exit | died | kills | doors | area (128u cells) |",
                "|---|---|---|---|---|---|"]
        rep += [f"| {x['level']} | {x['exit']} | {x['dead']} | {x['kills']} | {x['doors']} | {x['cells_128']} |"
                for x in real["levels"]]
    rep += ["", "## Videos"] + vids
    (OUT / "FINAL_REPORT_dry.md" if a.dry else ROOT / "runs" / "FINAL_REPORT.md").write_text("\n".join(rep) + "\n")
    log("\n".join(rep))
    if not a.dry:
        notify(f"Final: {best_c.name} score {best_s['weighted_score']:.0f}" + (" - pushed" if ok else " - NOT pushed"))


if __name__ == "__main__":
    main()
