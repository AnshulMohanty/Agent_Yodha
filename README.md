# YODHA: a DOOM agent that remembers, explores and finishes levels

**Yugaantar 2026 · DevForge, Track 1 "Can It Play DOOM?"**. Built from the official
[devforge-doom-template](https://github.com/Reinforce-SST/devforge-doom-template).

YODHA ("warrior") is a recurrent neural agent. It plays **unseen** DOOM levels using only the official
ego-centric observations: rays, entities, vitals and a small pixel view. It builds its own spatial memory as it plays.

| What makes it different | Why it matters for the score |
|---|---|
| **EgoMap memory.** A two-scale egocentric semantic map built from the 64 rays and the agent's own calibrated odometry (walls, doors, locked doors, keys, items, enemies, visited cells, frontier). | Remembers where the locked door was and which areas are explored, so it backtracks with a key instead of looping. |
| **Honest privileged teacher, then kickstarted PPO.** A training-only teacher sees the true map but only *knows* what the agent has observed. The student is DAgger-distilled from it, then improved with PPO. | Gets long-horizon key → locked door → exit behaviour in millions of steps instead of billions. |
| **Procedural maps in the organizers' own format.** UDMF grid maps with the same door, lock and exit specials as the template fixtures, plus real Freedoom 2 levels. | Trains for zero-shot generalisation across tiers, not memorisation. |
| **Entity set-attention + pixel CNN.** | Any number of enemies; sees exits (GATE floors, exit switches) that rays cannot. |
| **Intent head (hierarchical control).** explore / fight / get key / open door / go to exit / pickup, FiLM-conditioning the action heads. | Interpretable decisions (see the glass-box videos). |
| **Scorer-aligned reward.** Exit, keys, doors, unlocks, new sectors, kills, damage and death, computed from privileged state during training only. | Optimises the categories the judges score. |

## Architecture
```
obs ─┬─ rays (64×[dist, hit-type, Δdist]) ──► circular Conv1d ───────────────┐
     ├─ visible_entities (≤32) ─────────────► MLP + set-attention (2 queries) ┤
     ├─ vitals + self-history ──────────────► MLP                             ├─► Linear ─► GRU(512) ─┬─► intent head (6) ─┐
     ├─ rgb+depth 60×80 ────────────────────► CNN                             │                       │       FiLM ◄────────┘
     └─ odometry ─► EgoMap 320×320×9 world grid ─► fine 64×64 crop ─► CNN ────┤                       └─► 6 factorized action heads
                                                 coarse 32×32 crop ─► CNN ────┘                           (move/strafe/turn/attack/use/weapon → 9 buttons)
training only: value head with privileged critic features, auxiliary heads (exit in view, door ahead, frontier direction)
```
About 4.6M parameters, about 5 ms per decision on **one CPU thread** (budget 40 ms). `validate.py` passes.

## Repository
```
submission/agent.py         DoomAgent (evaluation entry point)
submission/yodha/           inference package: odometry, egomap, features, model, policy (numpy + torch only)
submission/weights/         yodha.pt (Git LFS), config.json, odometry.json
training/mapgen/            procedural tiered maps → UDMF WADs (organizer format) + solvability check
training/teacher/           honest privileged teacher (training only: grid BFS + aiming live here, never in submission/)
training/envs/              shaped-reward training env + shared-memory multiprocess workers
training/algo/              kickstarted recurrent PPO (our implementation)
training/calib/             odometry calibration
training/eval/harness.py    scorer replica on held-out maps (tier-weighted proxy score)
training/viz/glassbox.py    video: game + EgoMap + intent/action probabilities
training/cloud/             Kaggle runner
tests/                      official tests + mapgen, rules audit, agent contract
docs/                       contract/rules (official), TRACK1_MASTER_PLAN.md, FACTS.md, RUNBOOK.md
```

## Run it
```bash
uv sync --all-groups                      # Python 3.13, vizdoom 1.3.3, torch 2.15 (same as the official image)
uv run python validate.py                 # official checks
uv run pytest -q tests/                   # official + our tests
```
Training (Mac / any machine):
```bash
PYTHONPATH=. uv run python -m training.calib.odometry_calibrate                         # fit odometry (once)
PYTHONPATH=. uv run python -m training.teacher.run_teacher --maps 10                     # teacher sanity check
PYTHONPATH=. uv run python -m training.algo.kickstart_ppo --config training/configs/mac.yaml
PYTHONPATH=. uv run python -m training.eval.harness --weights runs/mac_main/export --maps 8
PYTHONPATH=. uv run python -m training.viz.glassbox --weights runs/mac_main/export --tier 3 --out runs/videos/t3.mp4
```

## Rules compliance
- Every action comes from YodhaNet's trained parameters (`submission/yodha/policy.py`, argmax or sampling over the heads).
- **No** rule trees, path planners or aimbots in `submission/`. `tests/test_audit.py` enforces it: no BFS/A*/Dijkstra/heapq/deque search, no atan2, no imports from `training/`.
- EgoMap and odometry are observation preprocessing plus memory. Nothing plans on them; they're inputs to the CNN.
- The privileged teacher, map knowledge and privileged reward are **training-time scaffolding only** (allowed by docs/RULES.md §1–2).
- PPO is our own implementation. Its implementation details follow CleanRL's `ppo.py` (MIT), used as a reference, and Stable-Baselines3 isn't used.
- Training maps are our own procedural maps plus Freedoom 2 (bundled with ViZDoom). Evaluation maps are never accessed.

## Credits and references
CleanRL PPO (reference); Beeching et al. *EgoMap* (2020) and *Deep RL on a Budget* (2019); Parisotto & Salakhutdinov *Neural Map*;
Lample & Chaplot *Playing FPS Games with Deep RL* (auxiliary game features); Chen et al. *Learning by Cheating* (privileged teacher);
Schmitt et al. *Kickstarting Deep RL*; Ross et al. *DAgger*. ViZDoom and Freedoom.

---
<details><summary>Original template instructions</summary>

See [docs/CONTRACT.md](docs/CONTRACT.md), [docs/RULES.md](docs/RULES.md), [docs/SUBMISSION.md](docs/SUBMISSION.md),
[docs/EVALUATION.md](docs/EVALUATION.md). Quickstart: `uv sync && uv run python validate.py`; Docker: `docker build -t devforge-doom .`
then `python validate.py --docker --image devforge-doom`.
</details>
