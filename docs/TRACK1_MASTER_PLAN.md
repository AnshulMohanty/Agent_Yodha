# TRACK 1 MASTER PLAN: "Can It Play DOOM?", agent YODHA

> **On approval, the executor does this first:**
> 1. Save this document as `/Users/anshulmohanty/Hackathon/TRACK1_MASTER_PLAN.md`.
> 2. Save the Track 2 master plan from this session as `/Users/anshulmohanty/Hackathon/TRACK2_MASTER_PLAN.md` for the teammate.
>
> Part A is the technical spec for the coding agent. Part B is the manual runbook for you, in simple language.

## Context
- You're doing **Track 1** solo with Claude Code, with **20 hours** of time. Your teammate is doing Track 2.
- **Hardware:** MacBook Air M5 (10 cores, 16 GB) plus free cloud.
- **Goal:** a neural agent that plays **unseen** DOOM maps zero-shot through the official `doom_env` wrapper. It should fight, explore, pick up keycards, open locked doors and reach the exit.
- **Scoring (hidden maps, 4 tiers, harder tiers weigh more):** exit (largest), progression (keys, doors, unlocks), exploration (unique areas), combat (kills minus damage and death).
- **Top teams** get a code and replay review for rule violations, then a desk demo. The final round rewards custom architectures: spatial memory, curiosity, hierarchy.

---

# PART A: Technical master plan

## A0. Facts from the official template (`Reinforce-SST/devforge-doom-template`), already verified

**Observation** (`env/doom_env/spaces.py`). All ego-centric; no global coordinates or heading.
- `rgb` `(120,160,3)` uint8, HUD hidden.
- `depth` `(120,160)` uint8.
- `vitals[17]`:
  - health, armor, weapon, ammo, 4 ammo types.
  - **6 key flags** (blue/yellow/red card, blue/yellow/red skull).
  - **vel_forward, vel_right, vel_up** in the player frame, map units per tic.
- `rays`: **64 rays over 360°**, ray 0 straight ahead, counter-clockwise, range 1024.
  - `distance/1024`.
  - `hit_type` ∈ {0 none, 1 wall, 2 door, 3 locked_door, 4 enemy, 5 item}.
- `visible_entities`: up to 32 entities, nearest first.
  - `features [dist/1024, sin, cos, rel_vel_fwd, rel_vel_right]`.
  - `entity_type` ∈ {pad, enemy_melee, enemy_hitscan, enemy_projectile, projectile, health, armor, ammo, weapon, key, barrel, other}.
  - `mask`.
  - Only entities in the 90° FOV and in line of sight.

**Action:** `MultiBinary(9)` = [FWD, BACK, STRAFE_L, STRAFE_R, TURN_L, TURN_R, ATTACK, USE, NEXT_WEAPON]. Each action is held for **4 tics**.
- **USE fires on the button-down edge.** Release it for ≥ 1 step before pressing again.
- **Pressing USE on a moving door reverses it.** Press once, then wait about 1–2 s (≈ 9–18 steps).

**Episode:** 2,100–4,200 tics = **525–1,050 decisions**. IWAD `freedoom2.wad`, skill 3, vizdoom **1.3.3**, vanilla rules.

**Agent API:** `submission/agent.py`, class `DoomAgent(observation_space, action_space, device)`, with `reset()` and `act(obs) -> int (9,) in {0,1}`.
- Load weights relative to `Path(__file__).parent`.
- The repo is read-only. `/tmp` only. No network.

**Eval sandbox:** **1 CPU thread** (`OMP_NUM_THREADS=1`), 8 GB RAM, 2 GB VRAM on an RTX 3090.
- Limits: init 60 s, reset 10 s, first act 10 s, any act 5 s, **compute budget 40 ms × steps per map**.
- A crash or timeout means the player **stands still** for the rest of that map.

**Base image:** Python **3.13**, `torch==2.15.0`, `numpy==2.5.5`, `gymnasium==1.4.0`, `vizdoom==1.3.3`, scipy, opencv-headless, jax/flax/optax, einops, tensorboard.
- Extra packages go in `submission/requirements.txt`, pinned, at most 50.
- Weights go in **Git LFS** (already set in `.gitattributes`). Repo ≤ 1 GB.

**Training-only helpers:**
- `env.privileged_state()`: x, y, z, angle, health, kills, items, damage, keys, dead, exit, `sector_floor`, `sector_ceiling`, `door_lock`.
- `env.sector_polygons()`: what the scorer uses for coverage.
- `make_env(wad, map, episode_ticks=..., render_buffers=False)`.

**Validator:** `uv run python validate.py [--map x.wad:MAP01]` checks size, LFS, imports, interface, a full episode, latency pinned to 1 core (Linux only) and RAM. `--docker` runs the same inside the official image; use Linux x86 for that.

**Map format the organizers use** (from `tests/fixtures/*.wad`). These are **tiny UDMF (`namespace = "zdoom"`) maps drawn from ASCII grids** with **64-unit cells**. The engine builds nodes itself.
- **Door:** its own sector with `heightceiling = 0`. Both faces get `special = 12` (Door_Raise), `arg1 = 16` (speed), `arg2 = 150` (delay), `playeruse`, `repeatspecial`.
- **Locked door:** `special = 13` (Door_LockedRaise), `arg3` = lock (**129 red, 130 blue, 131 yellow**).
- **Exit:** a cell sector with floor **`GATE3`**, with `special = 243` (Exit_Normal) and `playercross` on its boundary lines. **Rays don't see the exit**; only pixels show the GATE floor.
- **Things:** 1 player, 3004 zombieman, 9 shotgun guy, 65 chaingunner, 3001 imp, 13 red card, 5 blue card, 6 yellow card, 2011 stimpack.
- **Legend used by their tests:** `# . P D R B E z c i h r`.
- **Big insight:** the hidden maps for at least some tiers are likely made by the same kind of generator. **We build our own generator in this exact format**, which puts our training data close to the eval distribution.

## A1. Rules compliance (your three repos and our design)
| Item | Allowed? | How we use it |
|---|---|---|
| CleanRL `ppo.py` (MIT) | ✔ as a reference | We **write our own** recurrent PPO with kickstarting. CleanRL is cited and its implementation details are followed, but the code isn't pasted verbatim, and nothing uses SB3. |
| nicknochnack/DoomReinforcementLearning (SB3 PPO on pixels) | ⚠ SB3 policies are banned at eval | **Lessons only:** reward shaping from damage, ammo and kill deltas; curriculum by difficulty; frame-skip 4 |
| swedishembedded/brain doom sample (Apache-2.0) | ⚠ Its engine-side **BFS route** would be banned for us | **Lessons only:** (1) search-then-compress (teacher, then distill); (2) memory of seen objects stored as positions and reported as bearings; (3) feeding the agent its **own history** (visits, stuck counter) mattered a lot; (4) train on generated maps and keep real maps as the exam; (5) audit the observation for "unfair" information |
| Privileged teacher (BFS/A* on our grid, atan2 aiming) | ✔ **Training only** ("any training-time scaffolding" is allowed) | It lives in `training/teacher/` and is **never imported by `submission/`**. The final actions come from the network. |
| EgoMap (ray projection plus odometry) | ✔ "Preprocessing/encoding observations" and "learned memory modules" | **No search or planning on the map at eval.** It's only an input to the CNN. Frontier detection is a local 3×3 convolution, not BFS. |
| Reward from `privileged_state()` | ✔ Explicitly allowed for shaping | Training only |
| `act()` exception fallback returns zeros (stand still) | ✔ | A safety net, not a decision rule |
| Hand `if/else` deciding actions, an atan2 aimbot, A*/BFS in `submission/` | ✘ | Never. Add an audit grep (§A12). |

---

## A2. What we build: YODHA in one paragraph
YODHA is a **recurrent, multi-modal policy network**. It reads rays, entities, vitals and a small pixel view, plus a **two-scale egocentric semantic map (EgoMap)** that it builds itself from the rays and its own odometry. It's trained in two steps:
1. **Kickstarted distillation** from an **honest privileged teacher** on procedurally generated maps in the organizers' format. "Honest" means the teacher only knows what the agent has already seen.
2. **PPO fine-tuning** with a reward that mirrors the official scorer, computed from privileged state, plus an asymmetric critic.

Extras: an auxiliary **intent head** (explore / fight / get key / open door / go exit / pickup) gives hierarchical, interpretable control, and a "glass-box" viewer shows the map, intent and action probabilities live for the demo.

## A3. Repo layout (inside your template-created repo)
```
submission/
  agent.py                 # DoomAgent: features → EgoMap → YodhaNet → action (CPU, 1 thread)
  yodha/                   # inference-only package: numpy + torch ONLY
    features.py            # obs dict → normalized tensors
    odometry.py            # calibrated heading/position integration (+ optional ray scan-match)
    egomap.py              # world-anchored semantic grid → 2 egocentric crops
    model.py               # YodhaNet (encoders, GRU, heads)
    policy.py              # stateful wrapper: hidden state, prev action, sampling w/ seeded RNG
  weights/yodha.pt (LFS)  weights/config.json  weights/odometry.json
  requirements.txt         # stays empty (everything is in the base image)
training/
  README.md
  mapgen/  grid.py tiers.py udmf.py textures.py solvable.py   # procedural maps in organizer format
  teacher/ knowledge.py gridnav.py combat.py teacher.py        # honest privileged teacher (BFS here only)
  envs/    shaped_env.py worker.py vec.py                      # reward shaping + multiprocess workers
  algo/    kickstart_ppo.py storage.py schedules.py            # own recurrent PPO + distillation
  calib/   odometry_calibrate.py
  eval/    harness.py proxy_score.py heldout_maps.py           # scorer replica via DoomAgent interface
  cloud/   modal_app.py                                        # Modal: 32-core container (+T4)
  viz/     glassbox.py record_video.py
  configs/ dev.yaml mac.yaml main.yaml
brief/ brief.html  figures/  brief.pdf
tests/ (official) + test_mapgen.py test_egomap.py test_agent_contract.py test_latency.py test_audit.py
```

## A4. Procedural map generator (`training/mapgen/`), built H1–H3
- **Grid of 64-unit cells**, written as UDMF `zdoom` exactly like the fixtures. One WAD per map, `MAP01`, generated **on the fly in each worker** in milliseconds into `/tmp`, so map diversity is unlimited.
- **Legend:**
  - Structure: `# . P D R B Y E X`, where `X` is an exit switch wall: one-sided line, special 243, `playeruse`, texture `SW1EXIT`.
  - Keys: `r b y` = 13, 5, 6; skulls optional.
  - Monsters: `z s c i d S k o l` = 3004, 9, 65, 3001, 3002 demon, 58 spectre, 69 hell knight, 3005 cacodemon, 3006 lost soul.
  - Items: `h m a A u v g G x` = 2011, 2012 medikit, 2018, 2019 armor, 2007 clip, 2008 shells, 2001 shotgun, 2002 chaingun, 2035 barrel.
- **Layout:**
  1. Random rooms (3×3 to 10×10 cells), joined by an MST plus 20% extra loops through 1–2-wide corridors. Add some dead-end branches.
  2. Doors (`D`) on 40% of room-corridor junctions.
  3. Put the exit room at the largest graph distance from the start.
- **Progression (T3/T4):**
  1. Pick a gate edge on the start→exit path and make it a locked door of colour c.
  2. Place key c in a room reachable *without* crossing the gate, preferably in a **side branch**, so the agent has to go there and come back.
  3. T4 chains 2–3 locks, with key₂ behind lock₁.
- **Population by tier:**

| Tier | Grid | Rooms | Monsters | Items | Locks |
|---|---|---|---|---|---|
| T1 combat | 12–20 | 1–3 | 4–10 mixed | medium | 0 |
| T2 explore | 20–36 | 5–12 | 2–6 | medium | 0 (doors) |
| T3 progression | 24–40 | 6–12 | 3–8 | medium | 1–2 |
| T4 gauntlet | 28–48 | 8–16 | 8–20, ambush closets behind doors | scarce | 2–3 |

- **Writer (`udmf.py`):**
  - Flood-fill floor regions separated by door and exit cells into sectors. Each door is its own sector with ceiling 0, and locked doors use `DOORRED`/`DOORBLU`/`DOORYEL` textures.
  - Emit linedefs on cell edges where the class changes. **Merge collinear edges**, since fewer lines means faster sensors.
  - The front side must face the sector on the right of v1→v2.
- **Domain randomization:** wall, floor and ceiling textures (Doom 2 names, which Freedoom provides), light levels 128–255, exit type (GATE1–4 floor or exit switch), start angle, ceiling heights 96–160, and optional ±16 floor steps.
- **`solvable.py`:** BFS over (cell, keys held) to verify that keys, locks and exit are reachable. Reject unsolvable maps.
- **Held-out set:** 25 fixed-seed maps per tier, **never used in training**, plus `maps/two_rooms.wad`, the template fixtures and Freedoom2 MAP01–05 (tick cap 4200).
- **Test:** generate 100 maps per tier; all load in `make_env`; `observation_space.contains(obs)`; solvable; the generator reproduces the fixtures' look.

## A5. Shaped training env and workers (`training/envs/`), built H2–H4
- **`ShapedEnv`** wraps `make_env(...)`, tracks `privileged_state()` each step, and returns:
  - `obs`, **EgoMap crops built from estimated odometry** (never privileged pose, so train and eval match), shaped reward, `teacher_action` and `teacher_intent` (if the teacher is enabled), `privileged_features` for the critic, and episode stats.
- **Reward** (it mirrors the scorer; all coefficients live in config):

| Event (detected from privileged state) | Reward |
|---|---|
| Exit reached | +10, plus 2×health/100, plus 2×(1 − t/T) |
| Key picked up (vitals key flag 0→1) | +3 |
| Door opened (door sector ceiling rises above floor+56) | +1 the first time per door |
| Locked door opened | +4 |
| New coverage cell (128-unit cell of privileged x,y) | +0.05; also +0.3 per new sector (point-in-polygon on `sector_polygons`) |
| Kill | +1 |
| Damage taken | −0.01 per HP |
| Death | −5 |
| Time | −0.001 per step |
| No displacement for 8 steps | −0.02 per step |
| USE pressed with no door/locked ray within 100 units ahead | −0.02 |
| Potential shaping | γΦ(s′) − Φ(s), with Φ = −0.002·BFS distance to the current *known* objective (teacher knowledge) |

- **Workers (`vec.py`):** multiprocessing pipes. Each worker runs N envs.
  - Load a new random map every 2 episodes. Env init is about 0.5 s, so measure it.
  - Use `render_buffers=True`, because the pixel branch needs it. Downsample in the worker to 60×80 RGB plus depth.
- **Throughput target:** measure it in H1. Expect very roughly 300–800 steps/s per core, so about 3–5k steps/s on the Mac and about 10–15k on a 32-core Modal box.

## A6. Odometry and EgoMap (`submission/yodha/`, shared with training), built H4–H6

### Odometry calibration (`training/calib/odometry_calibrate.py`)
- Run random and teacher actions on generated maps and log (actions, vitals velocities) against privileged (x, y, angle).
- **Heading:** Doom keyboard turning is deterministic (slow turn for the first held tics, then fast). Fit a table `Δθ[turn_button, consecutive_steps_held]`, and save it to `weights/odometry.json`.
- **Translation:** `Δp = R(θ)·k·(v_fwd, v_right)`, with k fitted by least squares (≈ 4 tics; try the average of previous and current velocity).
- **Target:** heading error < 2° and position drift < 5% of path length over 500 steps. Report drift curves for the brief.
- **Optional:** ray scan-matching correction. Use the circular shift of the 64 ray distances, with sub-ray parabola refinement, to correct heading drift.

### EgoMap (`egomap.py`, numpy, < 2 ms per step on 1 thread)
- A world-anchored grid of 256×256 cells at 32 units, with the origin at the start pose.
- **Channels (uint8):**
  0. Free, along each ray up to its hit.
  1. Wall.
  2. Door.
  3. Locked door.
  4. Enemy last seen (decays −16 per step).
  5. Item seen.
  6. Key seen (cleared near the agent when a key flag flips).
  7. Visited (saturating count).
  8. Projectile (decays fast).
- Entities are placed from (distance, sin, cos) plus the pose.
- **Outputs:** two **egocentric** (heading-up) crops produced by nearest-neighbour index mapping:
  - **fine** 64×64 at 32 u (±1024 u, matching ray range);
  - **coarse** 32×32 at 128 u (±2048 u, via block-max), which provides long-range memory of "the locked door I saw 1,500 units back".
- **Frontier** channel at crop time: free cells next to unknown, via a 3×3 max-pool. Local filtering only.
- **Agent-history scalars** (the brain-repo lesson): visits to the current cell, steps since the last new cell, stuck counter, steps since USE, previous action (9).

## A7. Honest privileged teacher (`training/teacher/`, training only), built H3–H6
- **Knowledge (honesty):** a per-episode "seen" grid built with privileged pose and the same rays, plus FOV line-of-sight for floor cells. **Exit cells count as known only once they've been in the 90° FOV with LOS**, because the student sees GATE floors in pixels. Keys and items count once they appear in `visible_entities`.
  - This avoids the "teacher knows things the student can't" problem: the student can learn every teacher decision from its history and the EgoMap.
- **Navigation:** BFS over known-free and frontier cells (`gridnav.py`), recomputed every 4 steps or whenever knowledge changes. Waypoint following uses privileged heading: turn until within 10°, then move, with small strafes to un-stick.
- **Intent selection, in priority order:**
  1. **FIGHT** if an enemy is visible within 900 u. Aim using entity sin/cos; ATTACK when |angle| < 6°; strafe alternately; switch weapon if ammo is 0; dodge a projectile closing in by strafing perpendicular.
  2. **PICKUP** if health < 40 and a known health item is ≤ 6 cells away.
  3. **GO_EXIT** if the exit is known and reachable with the keys held.
  4. **GET_KEY** if a known locked door blocks the frontier or the exit and its key is known.
  5. **OPEN_DOOR:** when the next waypoint is a door or locked door, face it, press USE once, wait up to 25 steps for the door ray to clear, then retry.
  6. **EXPLORE:** go to the nearest frontier, preferring unvisited rooms.
- **Teacher outputs:** the factorized action (§A8) and the intent label.
- **Gate:** teacher success on held-out maps, roughly **T1 kills ≥ 70%, T2 exit ≥ 70%, T3 exit ≥ 50%, T4 partial**. If it's weaker, still proceed; PPO improves on it.

## A8. YodhaNet (`submission/yodha/model.py`, about 2–3M parameters, < 15 ms per act on 1 CPU thread)
- **Ray encoder:** 64 rays × [dist, one-hot(6), Δdist from the previous step] → **circular** Conv1d ×3 (32→64→64 channels, circular padding) → flatten → 256.
- **Entity encoder (any number of enemies):**
  - Per entity: [5 features, type embedding (12→16)] → MLP 64.
  - **Masked multi-head attention pooling.** The query is a learned token plus the vitals embedding; output 128.
  - Also a "nearest threat" slot.
- **Vitals encoder:** health and armor /200, log ammo, 6 key flags, weapon one-hot, velocities → MLP 64.
- **Pixel encoder:** RGB plus depth at 60×80 → conv 8×8/4 (32), 4×4/2 (64), 3×3/1 (64) → 256. Its job is to see exits (GATE floor or exit switch), keys and door colours. *Optional ablation: drop it if throughput or latency is tight.*
- **EgoMap encoder:**
  - Fine: conv 5×5/2 (32), 3×3/2 (64), 3×3/2 (64) → 256.
  - Coarse: a similar but smaller stack → 128.
- **History scalars plus previous action** → 32.
- **Core:** concat (about 1,100) → Linear 512 → **GRU 512**.
- **Heads:**
  - **Intent head** (6 classes, supervised by the teacher) → softmax → FiLM-conditions the action heads. This is the hierarchical part.
  - **Factorized action heads:** move {none, fwd, back}, strafe {none, L, R}, turn {none, L, R}, attack {0,1}, use {0,1}, next_weapon {0,1}. Map them back to the 9 buttons.
  - **Value head, training only:** an asymmetric critic. Its input is the GRU state plus **privileged features**: BFS distance to the objective, keys needed, explored fraction, enemies alive. This branch is dropped at export.
  - **Auxiliary heads, training only:** predict the intent; whether the exit is in view; the direction of the nearest frontier (8 bins); whether a door is ahead. Arnold showed these "game feature" auxiliary tasks speed up learning.
- **Inference in `act()`:**
  - Sample with **temperature 0.7**, using an RNG seeded in `reset()`, or use greedy. Choose by eval.
  - The **previous action is an input**, so the network learns USE edge timing itself. No hand rules.
- **Export:** `state_dict` without the critic and auxiliary heads, saved with **torch 2.15** (same as the image). `torch.set_num_threads(1)`. Use CPU even when `device == "cuda"`, unless eval shows a clear gain; CPU is deterministic and avoids CUDA init time.

## A9. Training algorithm: kickstarted recurrent PPO (`training/algo/kickstart_ppo.py`, our own code)
- **Rollouts:** E envs × T = 128 steps; TBPTT chunks of 64 with hidden states stored at chunk starts; episode boundaries masked.
- **Loss:** `L = L_PPO + λ_ks·Σ_heads CE(π_student, a_teacher) + 0.5·CE(intent) + 0.2·L_aux + c_v·L_value − c_ent·H`.
  - Kickstarting is Schmitt et al. 2018; DAgger coverage comes for free because the **student acts** and the teacher labels every visited state.
- **Phase schedule:**

| Phase | Steps | λ_ks | PPO coef | Teacher-executes prob β | Notes |
|---|---|---|---|---|---|
| A: DAgger warm-up | 0–8M | 1.0 | 0 | 0.5 → 0 | Pure imitation; β mixing reaches deep states (keys, locks) early. PPO loss is masked on teacher-executed steps. |
| B: Kickstart | 8–30M | 1.0 → 0.1 | 1.0 | 0 | RL starts taking over |
| C: PPO fine-tune | 30M+ | 0.05 | 1.0 | 0 | Beat the teacher. Also Freedoom maps, which have no teacher labels. |

- **PPO hyperparameters** (CleanRL / "37 details"): γ 0.995, GAE λ 0.95, clip 0.2, 4 epochs, 8 minibatches, adv-norm on, vf-coef 0.5, max-grad 0.5, lr 3e-4 with linear decay, entropy 0.01 → 0.003.
- **Auto-curriculum:** sample a tier with probability ∝ learning progress (success in [0.2, 0.8] gets weight). Training-only "mods": more ammo and health early, monster count ramping up. Eval stays vanilla.
- **Checkpoints:** every 2M steps, kept on the Modal Volume or `runs/`. Log to TensorBoard: per-tier exit rate, keys, doors, unlocks, coverage, kills, deaths, teacher agreement, intent accuracy, steps/s.

## A10. Evaluation harness, a scorer replica (`training/eval/harness.py`)
- Plays held-out maps through **the real `DoomAgent` class**, which catches interface bugs, with `OMP_NUM_THREADS=1` and per-act timing.
- **Per-map proxy score:**
  - Exit: 50, plus 5·health/100, plus 5·(1 − t/T).
  - Keys: 10 each. Doors opened: 3 each. Locked doors unlocked: 8 each.
  - Coverage: 20 × (sectors visited / total sectors).
  - Combat: 2 per kill, −0.02 per damage, −10 for dying before 50% of the cap.
  - Tier weights 1, 2, 3, 4.
- Outputs a CSV and a markdown table for: random agent, teacher, each checkpoint, and ablations (no EgoMap, no pixels).
- **Selection rule:** highest total tier-weighted proxy score; ties go to the lower death rate.

## A11. Cloud and compute plan
| Resource | Use | Notes |
|---|---|---|
| **Mac M5** | Development, generator/teacher debugging, the first DAgger run (8 workers), eval harness, glass-box viewer | `uv sync` works on Apple Silicon (macOS 15+). `caffeinate -dimsu` during runs. |
| ~~Modal~~ | **Dropped.** It requires billing details. | — |
| **Kaggle** (no card; phone verification) | **PPO run #2** (Phase B/C), started from the Mac's Phase-A checkpoint with a different seed or entropy. GPU notebook: about 4 CPUs plus a T4 learner, 12h per session. Use **Save Version → Run All** so it runs in the background. Setup cell: `pip install uv` → `uv venv -p 3.13` → `uv pip install vizdoom==1.3.3 gymnasium numpy torch==2.15.0 -e env`. | Internet must be ON (needs phone verification). Checkpoints are written to `/kaggle/working`. |
| **GitHub Codespaces** (120 core-h/month, no card) | **PPO run #3, or the no-pixels ablation**, plus **`validate.py --docker`** on Linux x86 (the real 1-core pinning). Set the idle timeout to 240 min; the training loop prints progress every minute, and terminal output resets the idle timer. | 4-core ≈ 30h of quota. Use an 8-core machine if your account offers it (≈ 15h). |
| **Colab free** (no card) | Eval sweeps only (2 vCPU) | Disconnects when idle |
| **Teammate's laptop** (optional) | One extra seed if their Track 2 training is idle | — |
| **Checkpoint relay** | The Mac's Phase-A checkpoint is shared with `gh release create ckpt-A --prerelease runs/phaseA.pt`. Kaggle and Codespaces download it with `gh release download ckpt-A` or `wget`. Each platform's best checkpoint comes back the same way. The eval harness on the Mac picks the winner. | Release assets are public, and the weights go public at submission anyway |

## A12. Submission packaging (`submission/agent.py`)
- **`__init__`:**
  1. `torch.set_num_threads(1)`.
  2. Load `weights/yodha.pt`, `config.json` and `odometry.json`.
  3. Build the net and run **one dummy forward** (warm-up).
- **`reset`:** zero the GRU state, the EgoMap, the odometry and the history; reseed the RNG.
- **`act`:**
  1. Features → EgoMap update → net forward → sample → 9-button int8 array.
  2. Wrap everything in try/except and return zeros on error. **Never print** per step.
- **Audit test (`tests/test_audit.py`):**
  - `submission/` must not import `training/`.
  - `submission/` must contain no `atan2`, `heapq`, `deque`-BFS, `astar` or `dijkstra`.
- **LFS:** `git lfs ls-files` lists `submission/weights/yodha.pt`. Repo size is small.
- **Validator:** `uv run python validate.py` on the Mac, and `--docker` on Codespaces, with `--map` pointing at 3 generated held-out maps. Must pass, with act p99 < 25 ms.

## A13. Demo, replays and brief
- **Glass-box viewer** (`training/viz/glassbox.py`, OpenCV): game frame, EgoMap crops (fine and coarse), intent probabilities, action probabilities, attention weights over entities, and odometry against the privileged trace (training maps only). Record MP4s with `record_video.py`. `env.reset(options={"record_path": ...})` saves `.lmp` demos for replay.
- **Brief (2 pages, `brief/brief.html` → headless Chrome → PDF):**
  - **Page 1:** USP, architecture diagram, EgoMap design with drift plot, entity attention, intent head.
  - **Page 2:** generator plus honest teacher, the kickstart-to-PPO pipeline and reward table, results table (random / teacher / YODHA / ablations per tier), a rules-compliance statement, and limits.

## A14. Why YODHA wins
1. **Sample efficiency on a laptop budget.** Pure RL needs huge numbers of frames for keycard backtracking. A **privileged teacher plus kickstarting** gets competent long-horizon behaviour in a few million steps (Learning by Cheating; Kickstarting). Teams running plain PPO will mostly score combat and exploration points only.
2. **Data matched to the eval distribution.** Our generator writes the **same UDMF grid format as the organizers' own fixtures** (same specials, lock numbers, exit floor), with randomization for robustness.
3. **Spatial memory that actually remembers:**
   - The two-scale EgoMap keeps locked doors, keys and visited areas across rooms. That targets the heaviest-weighted Tiers 3 and 4 and also stops loops and spinning, which helps the exploration score.
   - This follows EgoMap and Neural Map, but we build it from **calibrated odometry plus rays**, which is cheap enough for 1 CPU thread.
4. **Variable enemies** are handled by set-attention, and the **pixel branch** spots exits the rays can't see.
5. **Hierarchical and interpretable.** The intent head is a hierarchical control policy judges can *see* working in the glass-box viewer.
6. **Scorer-aligned shaping:**
   - The reward mirrors each scoring category (exit, keys, doors, unlocks, sector coverage, kills, damage, death), using the scorer's own `sector_polygons`.
   - An asymmetric critic speeds up credit assignment.
7. **Zero-crash engineering.** About 10 ms per act on 1 thread, a warm-up in `__init__`, try/except, validator and Docker checks. A crash means standing still, which many teams will hit.

---

# PART B: Your manual runbook (simple language, 20 hours)
> **Legend:** 👤 = you do it by hand. 🤖 = tell Claude Code to do it (paste the prompt). ✅ = check before moving on.

## Phase 0 (AUTOMATED by Claude, about 20 min)
Repo: `AnshulMohanty/Agent_Yodha`. Right now it's public and holds only a README. It is **not** linked to the template.

🤖 Claude runs these steps itself:
1. Fix the template link (**user chose Option A**): `gh repo rename Agent_Yodha-old -R AnshulMohanty/Agent_Yodha --yes`, then `gh repo create AnshulMohanty/Agent_Yodha --public --template Reinforce-SST/devforge-doom-template`. Nothing is deleted.
2. `gh repo clone AnshulMohanty/Agent_Yodha /Users/anshulmohanty/Hackathon/Agent_Yodha`.
3. `git lfs install`, then `uv sync`. uv fetches Python 3.13.
4. `uv run python validate.py`: the random agent should pass.
5. Add `git remote add template https://github.com/Reinforce-SST/devforge-doom-template`, so organizer patches can be merged later.
6. Write `docs/TRACK1_MASTER_PLAN.md` and fill `/Users/anshulmohanty/Hackathon/TRACK1_MASTER_PLAN.md`. The latter is currently empty.
7. Measure steps/s and env init time, and write them to `docs/FACTS.md`.
8. Scaffold the §A3 folders, then commit and push.

👤 You only need to approve the actions and make the Kaggle phone verification.

## Phase 0 (old manual version, kept for reference): Accounts and setup (H0:00–H1:00)
1. 👤 **Create the repo.** Open `https://github.com/Reinforce-SST/devforge-doom-template` → **Use this template** → **Create a new repository** → name it (e.g. `yodha-doom`) → **Public** → Create.
2. 👤 **Install tools on the Mac:**
   ```bash
   brew install uv git-lfs gh
   git lfs install
   gh auth login
   ```
3. 👤 **Clone and test:**
   ```bash
   cd ~/Hackathon
   gh repo clone <you>/yodha-doom
   cd yodha-doom
   uv sync
   uv run python validate.py
   ```
   ✅ It should end with all OK for the random agent. uv downloads Python 3.13 by itself; ignore your system 3.15.
4. 👤 **Modal:**
   - Sign up at modal.com with GitHub.
   - Run `uv tool install modal` then `modal setup` (a browser window opens; approve).
   - In the dashboard, check that the free credits show up.
   - Never put the token in the repo.
5. 👤 **Codespaces:** on your repo page → **Code** → **Codespaces** → **Create** (4-core). Leave it stopped until Phase 5.
6. 👤 *(Optional)* **Kaggle:** sign up and verify your phone in Settings.
7. 👤 **Start Claude Code** in `~/Hackathon/yodha-doom` and copy `TRACK1_MASTER_PLAN.md` into the repo's `docs/`.
   - 🤖 Prompt: *"Read docs/TRACK1_MASTER_PLAN.md. Measure env steps/s on maps/two_rooms.wad and freedoom2 MAP01 with render_buffers on and off, and env init time. Write the numbers to docs/FACTS.md."*

## Phase 1: Map generator and training env (H1–H4)
- 🤖 *"Implement §A4 (map generator in the organizers' UDMF format) with tests, then §A5 (ShapedEnv + multiprocess workers + reward table)."*
- 👤 ✅ Ask Claude to print 3 ASCII maps per tier, and open 2 of them in the glass-box viewer (or just a frame dump). Check that walls, doors, keys and exit look right and that `make_env` loads them.
- 👤 ✅ Note the steps/s with 8 workers. Above about 2,000 on the Mac is good.

## Phase 2: Teacher, odometry, EgoMap, model, agent (H3–H7)
- 🤖 *"Implement §A7 honest teacher; run it on 25 held-out maps per tier with the eval harness (§A10) and report success."*
- 🤖 *"Implement §A6 odometry calibration and EgoMap; show the drift plot."*
- 🤖 *"Implement §A8 YodhaNet and §A12 agent.py with random weights; run validate.py and the latency test."*
- 👤 ✅ Watch the teacher play 3 maps in the glass-box viewer. It should walk to keys, open doors and reach exits.
- 👤 ✅ Odometry drift < 5%. act() latency < 15 ms. `validate.py` passes.

## Phase 3: First training on the Mac and the first real submission (H7–H9)
- 🤖 *"Implement §A9 kickstart PPO. Start Phase A (DAgger) on the Mac with configs/mac.yaml (8 workers)."*
- 👤 Plug in the charger and run `caffeinate -dimsu` in a separate terminal. Run `uv run tensorboard --logdir runs` and open `http://localhost:6006`.
- 👤 ✅ **What "good" looks like:** teacher-agreement rising above 70%, exit rate on T1/T2 rising, deaths falling.
- 🤖 At about 3–5M steps: *"Export the best checkpoint per §A12, run eval harness + validate.py, commit weights with LFS and push."*
- 👤 ✅ `git lfs ls-files` shows `yodha.pt`. The push succeeded. **You now have a safety submission.**

## Phase 4 (REVISED, no Modal): parallel runs (H9–H17)
- **Mac:** continues as PPO run #1, the main run. Keep it plugged in with `caffeinate`.
- **Kaggle:** PPO run #2 from the ckpt-A release. Use Save Version → Run All (background, 12h).
- **Codespaces:** PPO run #3, or the no-pixels ablation. Idle timeout 240 min.
- **Every 2h:**
  1. Pull each platform's newest checkpoint.
  2. 🤖 Run the eval harness on the Mac.
  3. Keep the best.

## Phase 4 (old Modal version, superseded): Main training on Modal, sleeping included (H9–H17)
- 🤖 *"Implement §A11 Modal app (32 CPU + T4, Volume checkpoints, --max-hours 9). Launch Phase A→B→C with configs/main.yaml in detached mode."*
- 👤 Run the command Claude gives you, something like `modal run --detach training/cloud/modal_app.py::train`. Open the Modal dashboard and check that the logs show steps/s and that cost per hour is around $2.
- 👤 Every 1–2 hours, before and after sleeping:
  1. Download the newest checkpoint: `modal volume get yodha-runs ckpt/latest.pt runs/`.
  2. 🤖 *"Run the eval harness on runs/latest.pt and compare with the previous best."*
- 👤 **Sleep about H11–H15.** Modal keeps running. Before sleeping, check the run is healthy and the hard stop is set.
- 👤 *(Optional)* Run the "no-pixels" ablation on the Mac at the same time, for the brief table.
- ✅ **If something goes wrong:**

| Symptom | Fix |
|---|---|
| Teacher agreement stuck below 40% | Lower the learning rate, check the features |
| Agent spins in place | Raise the stuck penalty, check the visited channel |
| USE spam | Check that the previous-action input is wired in |
| Credits running low | Stop early and use the latest best checkpoint |

## Phase 5: Pick the best checkpoint and harden it (H17–H18:30)
- 🤖 *"Eval the top 3 checkpoints on all held-out maps with greedy vs temperature 0.7; pick the best by proxy score; export; push."*
- 👤 Open the Codespace. In its terminal:
  ```bash
  git pull
  git lfs pull
  python validate.py --docker --map <3 generated maps>
  ```
  ✅ Everything passes. Act p99 < 25 ms on 1 core. RAM OK.
- 👤 ✅ Run `git status` and confirm it's clean. Then `git lfs ls-files`. The scorer takes **the last pushed commit on the default branch at the deadline**.

## Phase 6: Brief, demo video and submit (H18:30–H20)
- 🤖 *"Make the brief per §A13 (2 pages, figures from the eval CSV), and record 3 glass-box MP4s (T1 fight, T3 key→lock→exit, a T4 partial)."*
- 👤 Read the brief once and check that the numbers match the eval table. Export the PDF.
- 👤 Run the final `validate.py` and do the final push **at least 30 minutes before the deadline**.
- 👤 Submit the repo URL and the brief PDF on the hackathon form.
- 👤 Prepare for the desk demo:
  - Have the glass-box viewer ready on 2 maps.
  - Rehearse the 60-second pitch: teacher → kickstart → PPO; EgoMap memory; intent head; scorer-aligned reward.

## Throughout
- 👤 Commit and push after every phase, so there's always a working version online.
- 👤 Keep the Mac charging. Don't close the lid during Mac runs (Modal runs don't care).
- 👤 Never edit `env/` (the scorer ignores it). Never put Modal or GitHub tokens in files.
- 👤 If the organizers announce a template patch (a `CONTRACT_VERSION` bump), merge it from the template repo.

---

## Verification (end-to-end)
1. `uv run pytest -q`: official tests plus ours (mapgen loads and is solvable, EgoMap shapes and speed, agent contract, latency < 15 ms, audit with no planner or atan2 in `submission/`).
2. `uv run python validate.py --map <held-out T1/T2/T3>` on the Mac, and `python validate.py --docker` on Codespaces: all OK.
3. The eval harness table shows YODHA ≫ random and ≥ the teacher on T1/T2, with progress on T3/T4. Ablations show the EgoMap helps on T2–T4.
4. Clean clone of the pushed repo plus `git lfs pull`, then `validate.py` passes. The pushed commit equals the evaluated one.

## Sources and prior art
- Official template: https://github.com/Reinforce-SST/devforge-doom-template (CONTRACT, EVALUATION, RULES, SUBMISSION, wrapper, fixtures)
- CleanRL PPO: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
- brain doom sample: https://github.com/swedishembedded/brain/tree/main/samples/decision/doom
- nicknochnack: https://github.com/nicknochnack/DoomReinforcementLearning
- Beeching et al. 2019, *Deep RL on a Budget* (ViZDoom, consumer hardware in under 24h): https://arxiv.org/abs/1904.01806
- Beeching et al. 2020, *EgoMap*: https://arxiv.org/abs/2002.02286
- Parisotto & Salakhutdinov, *Neural Map*: https://arxiv.org/abs/1702.08360
- Lample & Chaplot, *Playing FPS Games with Deep RL* (game-feature auxiliary tasks): https://arxiv.org/abs/1609.05521
- Chen et al. 2019, *Learning by Cheating* (privileged teacher → student, DAgger): https://interactive-learning-algos.github.io/assets/pdfs/LearningByCheating.pdf
- Teacher–student asymmetry fix: https://arxiv.org/pdf/2412.09149v1
- Henaff et al. 2022, *E3B* episodic exploration: https://arxiv.org/abs/2210.05805
- Sample Factory (ViZDoom throughput): https://arxiv.org/abs/2006.11751
- Schmitt et al. 2018, *Kickstarting Deep RL*: https://arxiv.org/abs/1803.03835
