# Training pipeline

1. **Maps** (`mapgen/`): `generate(tier, seed)` builds a solvable grid layout, then `write_map` produces a UDMF WAD in
   the organizers' fixture format.

   | Tier | Contents |
   |---|---|
   | T1 | Combat arenas |
   | T2 | Multi-room with doors |
   | T3 | Key → locked door → exit, with the key in a side branch |
   | T4 | 2–3 chained locks, ambush closets, scarce items |

   Doors are never placed next to locks, because Doom doors open only to their lowest neighbouring ceiling.
2. **Odometry** (`calib/odometry_calibrate.py`): fits the heading table and translation matrix against privileged pose.
   Heading drift is < 1° over 500 steps; position drift is about 7% of path.
3. **Teacher** (`teacher/teacher.py`): honest privileged expert. It uses the true grid and pose, but only knows what the
   agent has seen (rays, exits only once inside the 90° FOV, keys only once in `visible_entities`). Its priorities are
   fight > low-health pickup > exit > key > nearest frontier, and it presses USE once per door.
4. **Env** (`envs/yodha_env.py`): one map is reused for 4 episodes, then a fresh one is generated. The reward mirrors the
   scoring categories:

   | Event | Reward |
   |---|---|
   | Exit | +10 (+ health/time bonus) |
   | Key | +3 |
   | Door | +1 |
   | Locked door | +4 |
   | New 128-unit cell | +0.05 |
   | New sector | +0.3 |
   | Kill | +1 |
   | Damage | −0.01 per HP |
   | Death | −5 |
   | Time | −0.001 per step |
   | Stuck / USE spam | small penalties |
   | Teacher-objective potential | small shaping term |

   A share of episodes (`real_map_prob`) uses real Freedoom 2 levels, with RL only and no teacher.
5. **Workers** (`envs/vec.py`): spawn processes write observations and labels to shared memory.
6. **Learner** (`algo/kickstart_ppo.py`): recurrent PPO with GAE, clipping and TBPTT over 32 steps, plus a kickstarting
   cross-entropy towards the teacher on every visited state (DAgger). Training runs in three phases:

   | Phase | Teacher weight | PPO | Who acts |
   |---|---|---|---|
   | A | 1.0 | off | The teacher executes for part of the time (β 0.5 → 0) |
   | B | 1.0 → 0.1 | on | The student |
   | C | 0.05 | on | The student |

   A tier curriculum favours tiers with a 20–80% success rate.
7. **Evaluation** (`eval/harness.py`): held-out seeds, played through `YodhaPolicy` (the same code path as `DoomAgent`).
   It reports a tier-weighted proxy score.

Logs: `runs/<run>/tb` (TensorBoard) and `runs/<run>.log`. Checkpoints: `runs/<run>/latest.pt`, `ckpt_XXXXM.pt`,
`export/` (inference weights).
