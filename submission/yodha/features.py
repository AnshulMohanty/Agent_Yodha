"""Observation → network inputs. Shared by training and the submitted agent (numpy only).

`AgentMemory` owns everything stateful the agent keeps between steps: odometry, EgoMap, previous action
and simple self-history counters (the agent's own history is a strong cue for not looping/spinning).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .egomap import CROP_CH, EgoMap
from .odometry import Odometry

NUM_RAYS, RAY_TYPES = 64, 6
MAX_ENT, ENT_FEAT, NUM_ENT_TYPES = 32, 5, 12
SCALAR_DIM = 46
PIX_H, PIX_W = 60, 80


def scalar_features(vitals: np.ndarray, prev_action: np.ndarray, hist: np.ndarray) -> np.ndarray:
    v = vitals.astype(np.float32)
    weapon = np.zeros(8, np.float32)
    weapon[int(np.clip(v[2], 0, 7))] = 1.0
    out = np.concatenate([
        [v[0] / 100.0, v[1] / 100.0, math.log1p(max(v[3], 0)) / 5.0],
        np.log1p(np.maximum(v[4:8], 0)) / 5.0,
        weapon,
        v[8:14],                      # keys held
        v[14:17] / 10.0,              # ego velocity
        prev_action.astype(np.float32),
        hist.astype(np.float32),      # self-history counters (see AgentMemory)
    ])
    return out.astype(np.float32)


class AgentMemory:
    HIST_DIM = 13  # = SCALAR_DIM - (3 + 4 + 8 + 6 + 3 + 9)

    def __init__(self, odometry_calib: str | Path | dict | None):
        self.odo = Odometry(odometry_calib)
        self.map = EgoMap()
        self.reset()

    def reset(self) -> None:
        self.odo.reset()
        self.map.reset()
        self.prev_action = np.zeros(9, np.int8)
        self.prev_dist = None
        self.step = 0
        self.stuck = 0
        self.since_new_cell = 0
        self.since_use = 50
        self.since_damage = 50
        self.prev_health = None
        self.prev_pos = (0.0, 0.0)

    def observe(self, obs: dict) -> dict:
        """Update memory with the observation that resulted from `self.prev_action`; return net inputs."""
        if self.step > 0:
            x, y, h = self.odo.update(self.prev_action, obs["vitals"])
        else:
            x, y, h = 0.0, 0.0, 0.0
        new_cell = self.map.visits_here(x, y) == 0
        self.map.update(obs, x, y, h)
        moved = math.hypot(x - self.prev_pos[0], y - self.prev_pos[1])
        self.stuck = self.stuck + 1 if moved < 2.0 else 0
        self.since_new_cell = 0 if new_cell else self.since_new_cell + 1
        self.since_use = 0 if self.prev_action[7] else self.since_use + 1
        hp = float(obs["vitals"][0])
        if self.prev_health is not None and hp < self.prev_health:
            self.since_damage = 0
        else:
            self.since_damage += 1
        self.prev_health = hp
        self.prev_pos = (x, y)

        dist = obs["rays"]["distance"].astype(np.float32)
        ddist = np.zeros_like(dist) if self.prev_dist is None else dist - self.prev_dist
        self.prev_dist = dist
        onehot = np.eye(RAY_TYPES, dtype=np.float32)[obs["rays"]["hit_type"].astype(np.int64)]
        rays = np.concatenate([dist[:, None], onehot, np.clip(ddist, -0.2, 0.2)[:, None] * 5.0], 1)  # (64, 8)

        visits = self.map.visits_here(x, y) / 255.0
        hist = np.array([
            visits, min(self.stuck, 20) / 20.0, min(self.since_new_cell, 100) / 100.0,
            min(self.since_use, 30) / 30.0, min(self.since_damage, 30) / 30.0, min(self.step, 1050) / 1050.0,
            math.sin(math.radians(h)), math.cos(math.radians(h)), np.tanh(x / 2048.0), np.tanh(y / 2048.0),
            float(self.step == 0), min(moved, 20.0) / 20.0, float(self.prev_action[7]),
        ], np.float32)
        fine, coarse = self.map.crops(x, y, h)
        rgb = obs["rgb"][::2, ::2]
        depth = obs["depth"][::2, ::2]
        pix = np.concatenate([rgb.transpose(2, 0, 1), depth[None]], 0)  # (4, 60, 80) uint8
        ent = obs["visible_entities"]
        self.step += 1
        return {
            "rays": rays,
            "ent_feat": ent["features"].astype(np.float32),
            "ent_type": ent["entity_type"].astype(np.int64),
            "ent_mask": ent["mask"].astype(np.float32),
            "scalars": scalar_features(obs["vitals"], self.prev_action, hist),
            "pix": np.ascontiguousarray(pix, dtype=np.uint8),
            "map_fine": fine,
            "map_coarse": coarse,
        }

    def record_action(self, action: np.ndarray) -> None:
        self.prev_action = np.asarray(action, np.int8).copy()


INPUT_SPECS = {
    "rays": ((NUM_RAYS, 8), np.float32),
    "ent_feat": ((MAX_ENT, ENT_FEAT), np.float32),
    "ent_type": ((MAX_ENT,), np.int64),
    "ent_mask": ((MAX_ENT,), np.float32),
    "scalars": ((SCALAR_DIM,), np.float32),
    "pix": ((4, PIX_H, PIX_W), np.uint8),
    "map_fine": ((CROP_CH, 64, 64), np.uint8),
    "map_coarse": ((CROP_CH, 32, 32), np.uint8),
}
