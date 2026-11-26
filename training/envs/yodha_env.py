"""Training environment: procedural map + official doom_env + agent memory + honest teacher + shaped reward.

The reward mirrors the official scoring categories using privileged state (allowed for training):
exit (+health/time bonus), keys, doors opened, locked doors unlocked, new areas/sectors, kills,
damage, death; plus small anti-stuck / anti-USE-spam terms and potential shaping toward the teacher's
current objective.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np

import vizdoom

from doom_env import make_env
from submission.yodha.features import AgentMemory
from training.mapgen.grid import generate
from training.mapgen.udmf import _sectors, write_map
from training.teacher.teacher import INTENT, Teacher

warnings.filterwarnings("ignore")
ODO = Path(__file__).resolve().parents[2] / "submission" / "weights" / "odometry.json"
PRIV_DIM = 16
FREEDOOM2 = os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad")
REAL_MAPS = [f"MAP{i:02d}" for i in range(1, 33)]
REWARD = dict(exit=10.0, exit_health=2.0, exit_time=2.0, key=3.0, door=1.0, lock=4.0, cell=0.05, sector=0.3,
              kill=1.0, damage=-0.02, death=-8.0, time=-0.001, stuck=-0.02, use_spam=-0.02, shaping=0.02)
# 27 Nov 05:30 survival push: damage -0.01 -> -0.02 per HP, death -5 -> -8. Before, a full health bar cost the
# same as one kill paid, so the policy traded health for kills and died mid-level on tier 4 (deaths flat ~57%).


class YodhaEnv:
    def __init__(self, env_id: int, seed: int = 0, tier_weights=(0.35, 0.35, 0.2, 0.1), teacher: bool = True,
                 workdir: str | None = None):
        self.env_id = env_id
        self.rng = np.random.default_rng([seed, env_id])
        self.tier_weights = np.array(tier_weights, np.float64)
        self.use_teacher = teacher
        self.workdir = Path(workdir or f"/tmp/yodha_env/{os.getpid()}_{env_id}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.memory = AgentMemory(ODO if ODO.exists() else None)
        self.env = None
        self.episode = 0
        self.map_uses = 0
        self.tier = 1
        self.real_map_prob = 0.0
        self.spec = None

    def set_tier_weights(self, w) -> None:
        self.tier_weights = np.asarray(w, np.float64)

    def set_options(self, opts: dict) -> None:
        self.real_map_prob = float(opts.get("real_map_prob", self.real_map_prob))

    # ------------------------------------------------------------------ episode
    EPISODES_PER_MAP = 4  # re-use a map (new monster RNG seed) to amortize engine start-up

    def reset(self):
        seed = int(self.rng.integers(0, 2**31 - 1))
        if self.env is None or self.map_uses >= self.EPISODES_PER_MAP:
            if self.env is not None:
                self.env.close()
            if self.rng.random() < self.real_map_prob:
                # real hand-made Freedoom 2 level (no teacher; RL only) for geometry beyond our grid maps
                self.spec = None
                self.env = make_env(FREEDOOM2, str(self.rng.choice(REAL_MAPS)), episode_ticks=4200)
                self.tier = 5
            else:
                tier = int(self.rng.choice(4, p=self.tier_weights / self.tier_weights.sum())) + 1
                self.spec = generate(tier, seed)
                path = write_map(self.spec, self.workdir / f"ep{self.episode % 2}.wad", self.rng)
                self.env = make_env(str(path), "MAP01", episode_ticks=self.spec.episode_ticks)
                self.tier = tier
            self.map_uses = 0
        obs, _ = self.env.reset(seed=seed % (2**31))
        self.map_uses += 1
        self.episode += 1
        tier = self.tier
        self.memory.reset()
        if self.spec is not None:
            self.sid, _ = _sectors(self.spec.grid)
            self.teacher = Teacher(self.spec, self.rng) if self.use_teacher else None
            self.n_monsters = sum(1 for t in self.spec.things if t.kind == "monster")
            self.floor_cells = max(1, int((self.spec.grid != "#").sum()))
            self.ticks = self.spec.episode_ticks
            self.n_locks = len(self.spec.locks)
            n_keys = len(self.spec.keys)
        else:
            self.sid, self.teacher = None, None
            self.n_monsters, self.floor_cells, self.ticks, self.n_locks, n_keys = 1, 2000, 4200, 0, 0
        st = self.env.privileged_state()
        self.prev_st = st
        self.cells_seen = {self._cell128(st)}
        self.sectors_seen = {self._sector(st)}
        self.doors_opened = set()
        self.stuck = 0
        self.ep_return = 0.0
        self.ep_stats = dict(tier=tier, exit=0.0, keys=0.0, n_keys=float(n_keys), doors=0.0, locks=0.0,
                             kills=0.0, monsters=float(self.n_monsters), dead=0.0, coverage=0.0, steps=0.0)
        self.obs = obs
        self.phi = 0.0
        inp = self.memory.observe(obs)
        return inp, self._extras(obs, st)

    def _cell128(self, st) -> tuple[int, int]:
        return int(st["x"] // 128), int(st["y"] // 128)

    def _sector(self, st) -> int:
        if self.sid is None:
            return -1
        r, c = int(st["y"] // 64), int(st["x"] // 64)
        if 0 <= r < self.sid.shape[0] and 0 <= c < self.sid.shape[1]:
            return int(self.sid[r, c])
        return -1

    # ------------------------------------------------------------------ teacher / labels
    def _extras(self, obs, st) -> dict:
        ex = {"priv": np.zeros(PRIV_DIM, np.float32), "teacher": np.zeros(9, np.int8), "intent": 0,
              "aux_exit": 0.0, "aux_door": 0.0, "aux_dir": -1, "has_teacher": float(self.teacher is not None)}
        if self.teacher is not None:
            a, intent = self.teacher.act(obs, st)
            ex["teacher"], ex["intent"] = a, intent
            ex["aux_exit"] = float(getattr(self.teacher, "exit_in_view", False))
            if self.teacher.target is not None:
                ty, tx = (self.teacher.target[0] + 0.5) * 64, (self.teacher.target[1] + 0.5) * 64
                rel = (math.degrees(math.atan2(ty - st["y"], tx - st["x"])) - st["angle"]) % 360
                ex["aux_dir"] = int(((rel + 22.5) % 360) // 45)
            self.phi = -0.0 if not self.teacher.path else -float(len(self.teacher.path))
        ht, hd = int(obs["rays"]["hit_type"][0]), float(obs["rays"]["distance"][0])
        ex["aux_door"] = float(ht in (2, 3) and hd < 0.2)
        p = ex["priv"]
        p[0] = st["health"] / 100.0
        p[1] = float((st["keys"] > 0).sum()) / 3.0
        p[2] = len(self.cells_seen) * 4.0 / self.floor_cells
        p[3] = min(-self.phi, 60.0) / 60.0
        p[4 + int(ex["intent"])] = 1.0
        p[10] = float(getattr(self.teacher, "exit_known", False))
        p[11] = (self.n_locks - sum(1 for _, lock in self.doors_opened if lock > 0)) / 3.0
        p[12] = st["kills"] / max(1, self.n_monsters)
        p[13] = st["tic"] / max(1, self.ticks)
        p[14] = self.tier / 4.0
        return ex

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        action = np.asarray(action, np.int8)
        prev, phi_prev = self.prev_st, self.phi
        obs, _, term, trunc, info = self.env.step(action)
        st = self.env.privileged_state()
        R = REWARD
        r = R["time"]
        if info.get("exit_reached"):
            r += R["exit"] + R["exit_health"] * max(st["health"], 0) / 100.0 + R["exit_time"] * (1 - st["tic"] / self.ticks)
            self.ep_stats["exit"] = 1.0
        new_keys = float((st["keys"] > 0).sum() - (prev["keys"] > 0).sum())
        if new_keys > 0:
            r += R["key"] * new_keys
            self.ep_stats["keys"] += new_keys
        if st["sector_ceiling"] is not None:
            for sec, lock in self.env.mapinfo.door_sectors.items():
                if (sec, lock) not in self.doors_opened and st["sector_ceiling"][sec] - st["sector_floor"][sec] > 56:
                    self.doors_opened.add((sec, lock))
                    r += R["lock"] if lock > 0 else R["door"]
                    self.ep_stats["locks" if lock > 0 else "doors"] += 1
        c = self._cell128(st)
        if c not in self.cells_seen:
            self.cells_seen.add(c)
            r += R["cell"]
        s = self._sector(st)
        if s >= 0 and s not in self.sectors_seen:
            self.sectors_seen.add(s)
            r += R["sector"]
        r += R["kill"] * (st["kills"] - prev["kills"])
        r += R["damage"] * max(0.0, st["damage_taken"] - prev["damage_taken"])
        if st["dead"]:
            r += R["death"]
            self.ep_stats["dead"] = 1.0
        moved = math.hypot(st["x"] - prev["x"], st["y"] - prev["y"])
        self.stuck = self.stuck + 1 if moved < 2.0 else 0
        if self.stuck > 8:
            r += R["stuck"]
        if action[7] and not (int(obs["rays"]["hit_type"][0]) in (1, 2, 3) and float(obs["rays"]["distance"][0]) < 0.15):
            r += R["use_spam"]

        done = bool(term or trunc)
        self.prev_st = st
        self.memory.record_action(action)
        if self.teacher is not None:
            self.teacher.prev_use = int(action[7])
        if done:
            self.ep_stats["kills"] = float(st["kills"])
            self.ep_stats["coverage"] = (len(self.sectors_seen) / max(1, int(self.sid.max()) + 1) if self.sid is not None
                                         else len(self.cells_seen) / 200.0)
            self.ep_stats["steps"] = float(self.memory.step)
            stats = dict(self.ep_stats)
            self.ep_return += r
            stats["return"] = self.ep_return
            inp, ex = self.reset()
            ex["episode"] = stats
            return inp, float(r), True, ex
        inp = self.memory.observe(obs)
        ex = self._extras(obs, st)
        r += R["shaping"] * (0.995 * self.phi - phi_prev)
        self.ep_return += r
        self.obs = obs
        return inp, float(r), False, ex

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
