"""Dead-reckoning odometry from the agent's own actions + ego-frame velocity in `vitals`.

No privileged information: heading comes from a calibrated table over the recent turn-button history
(Doom keyboard turning is deterministic), position from a calibrated linear model of the ego-frame
velocity. Pose is relative to the start of the map (origin, heading 0). Calibration constants live in
weights/odometry.json (fitted by training/calib/odometry_calibrate.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TURN_STEP = 14.0625  # degrees per decision while a turn button is held (4 tics x 3.515625)


def turn_of(action) -> int:
    return int(action[4]) - int(action[5])


def heading_key(step: int, hist: list[int], depth: int) -> str:
    """Key for the heading table. hist[0] is the current step's turn, hist[1] the previous one, ..."""
    if step < 2:
        return f"s{step}," + ",".join(str(t) for t in hist[: step + 1])
    return ",".join(str(t) for t in hist[:depth])


class Odometry:
    def __init__(self, calib: dict | str | Path | None = None):
        if isinstance(calib, (str, Path)):
            calib = json.loads(Path(calib).read_text())
        calib = calib or {}
        self.table: dict[str, float] = calib.get("turn_table", {})
        self.A = np.array(calib.get("trans_A", [[4.0, 0.0], [0.0, 4.0], [0.0, 0.0], [0.0, 0.0]]), np.float64)
        self.reset()

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # degrees, CCW positive, 0 = initial facing
        self.step = 0
        self.hist = [0, 0, 0, 0]
        self.prev_v = np.zeros(2)

    def predict_dtheta(self) -> float:
        for depth in (4, 3):
            key = heading_key(self.step, self.hist, depth)
            if key in self.table:
                return self.table[key]
        return 0.0 if self.step == 0 else TURN_STEP * self.hist[0]

    def update(self, action, vitals) -> tuple[float, float, float]:
        """Call once per decision AFTER `action` was executed and `vitals` is the resulting observation."""
        self.hist = [turn_of(action)] + self.hist[:3]
        dth = self.predict_dtheta()
        self.heading += dth
        v = np.asarray(vitals[14:16], np.float64)
        tr = math.radians(dth)
        pf, pl = self.prev_v[0], -self.prev_v[1]
        f2 = pf * math.cos(tr) + pl * math.sin(tr)
        l2 = -pf * math.sin(tr) + pl * math.cos(tr)
        d_fwd, d_right = np.array([v[0], v[1], f2, -l2]) @ self.A
        t = math.radians(self.heading)
        self.x += d_fwd * math.cos(t) + d_right * math.sin(t)
        self.y += d_fwd * math.sin(t) - d_right * math.cos(t)
        self.prev_v = v
        self.step += 1
        return self.x, self.y, self.heading
