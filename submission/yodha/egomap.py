"""EgoMap: a two-scale egocentric semantic memory built only from the agent's own observations.

Rays and visible entities are projected into a world-anchored grid using the agent's dead-reckoned
pose (odometry.py). Each step, two heading-up crops centred on the agent are read out for the policy
network. This is observation preprocessing + memory; no planning or path search happens here.

World grid: SIZE x SIZE cells of RES map units, origin = start pose at the grid centre.
Channels (uint8):
  0 free  1 wall  2 door  3 locked door  4 enemy (decays)  5 item  6 key  7 visited  8 projectile (decays)
Crops add channel 9 = frontier (free cell next to never-seen space), computed locally.
"""

from __future__ import annotations

import math

import numpy as np

RES = 32.0
SIZE = 320
CENTER = SIZE // 2
RAY_RANGE = 1024.0
NUM_RAYS = 64
C_FREE, C_WALL, C_DOOR, C_LOCK, C_ENEMY, C_ITEM, C_KEY, C_VISIT, C_PROJ = range(9)
N_CH = 9
CROP_CH = N_CH + 1
FINE, FINE_RES = 64, 32.0      # 64x64 cells of 32 units (±1024)
COARSE, COARSE_RES = 32, 128.0  # 32x32 cells of 128 units (±2048)

# entity type ids (doom_env.spaces.ENTITY_TYPES order)
ENEMY_TYPES = (1, 2, 3)
PROJ_TYPE = 4
ITEM_TYPES = (5, 6, 7, 8, 11)
KEY_TYPE = 9

_RAY_ANG = np.arange(NUM_RAYS) * (2 * math.pi / NUM_RAYS)
_SAMPLES = np.arange(8.0, RAY_RANGE, 16.0)  # sample distances along a ray


def _crop_offsets(n: int, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Ego offsets (forward, left) in map units for each crop pixel. Row 0 = far ahead, centre = agent."""
    idx = (np.arange(n) - n / 2 + 0.5) * res
    fwd = -idx[:, None] * np.ones((1, n))        # rows: top = ahead
    left = -idx[None, :] * np.ones((n, 1))       # cols: left side of the image = agent's left
    return fwd, left


_FINE_OFF = _crop_offsets(FINE, FINE_RES)
_COARSE_OFF = _crop_offsets(COARSE, COARSE_RES)


class EgoMap:
    def __init__(self):
        self.grid = np.zeros((N_CH, SIZE, SIZE), np.uint8)
        self.reset()

    def reset(self) -> None:
        self.grid[:] = 0
        self.prev_keys = np.zeros(6, np.float32)

    def _cell(self, x, y):
        return (np.floor(np.asarray(y) / RES).astype(np.int64) + CENTER,
                np.floor(np.asarray(x) / RES).astype(np.int64) + CENTER)

    def _put(self, ch: int, rows, cols, value: int = 255) -> None:
        ok = (rows >= 0) & (rows < SIZE) & (cols >= 0) & (cols < SIZE)
        self.grid[ch, rows[ok], cols[ok]] = value

    def update(self, obs: dict, x: float, y: float, heading_deg: float) -> None:
        g = self.grid
        # decays
        np.subtract(g[C_ENEMY], np.minimum(g[C_ENEMY], 24), out=g[C_ENEMY])
        np.subtract(g[C_PROJ], np.minimum(g[C_PROJ], 128), out=g[C_PROJ])

        h = math.radians(heading_deg)
        dist = obs["rays"]["distance"].astype(np.float64) * RAY_RANGE
        hit = obs["rays"]["hit_type"]
        ang = h + _RAY_ANG
        ca, sa = np.cos(ang), np.sin(ang)

        # free space along each ray (strictly before the hit)
        mask = _SAMPLES[None, :] < (dist[:, None] - 12.0)
        px = x + ca[:, None] * _SAMPLES[None, :]
        py = y + sa[:, None] * _SAMPLES[None, :]
        r, c = self._cell(px[mask], py[mask])
        ok = (r >= 0) & (r < SIZE) & (c >= 0) & (c < SIZE)
        r, c = r[ok], c[ok]
        g[C_FREE, r, c] = 255
        g[C_WALL, r, c] = 0  # opened doors / drift corrections
        g[C_DOOR, r, c] = 0
        g[C_LOCK, r, c] = 0

        # hit points
        hx = x + ca * (dist + 8.0)
        hy = y + sa * (dist + 8.0)
        hr, hc = self._cell(hx, hy)
        for code, ch in ((1, C_WALL), (2, C_DOOR), (3, C_LOCK), (4, C_ENEMY), (5, C_ITEM)):
            sel = hit == code
            if sel.any():
                self._put(ch, hr[sel], hc[sel])

        # visible entities
        ent = obs["visible_entities"]
        m = ent["mask"].astype(bool)
        if m.any():
            f = ent["features"][m]
            et = ent["entity_type"][m]
            d = f[:, 0] * RAY_RANGE
            ea = h + np.arctan2(f[:, 1], f[:, 2])
            er, ec = self._cell(x + np.cos(ea) * d, y + np.sin(ea) * d)
            for types, ch in ((ENEMY_TYPES, C_ENEMY), ((PROJ_TYPE,), C_PROJ), (ITEM_TYPES, C_ITEM), ((KEY_TYPE,), C_KEY)):
                sel = np.isin(et, types)
                if sel.any():
                    self._put(ch, er[sel], ec[sel])

        # the agent's own cell
        ar, ac = self._cell(x, y)
        if 0 <= ar < SIZE and 0 <= ac < SIZE:
            g[C_VISIT, ar, ac] = min(255, int(g[C_VISIT, ar, ac]) + 40)
            g[C_FREE, ar, ac] = 255
            r0, r1, c0, c1 = max(ar - 1, 0), ar + 2, max(ac - 1, 0), ac + 2
            g[C_ITEM, r0:r1, c0:c1] = 0  # picked up or not there anymore
        keys = obs["vitals"][8:14]
        if (keys > self.prev_keys).any() and 0 <= ar < SIZE and 0 <= ac < SIZE:
            g[C_KEY, max(ar - 3, 0):ar + 4, max(ac - 3, 0):ac + 4] = 0
        self.prev_keys = keys.copy()

    def visits_here(self, x: float, y: float) -> int:
        r, c = self._cell(x, y)
        return int(self.grid[C_VISIT, r, c]) if 0 <= r < SIZE and 0 <= c < SIZE else 0

    def _sample(self, grid: np.ndarray, x: float, y: float, h: float, off, res: float) -> np.ndarray:
        fwd, left = off
        ch, sh = math.cos(h), math.sin(h)
        wx = x + fwd * ch - left * sh
        wy = y + fwd * sh + left * ch
        n_cells = grid.shape[-1]
        center = n_cells // 2
        rr = np.floor(wy / res).astype(np.int64) + center
        cc = np.floor(wx / res).astype(np.int64) + center
        ok = (rr >= 0) & (rr < n_cells) & (cc >= 0) & (cc < n_cells)
        out = np.zeros((grid.shape[0],) + fwd.shape, np.uint8)
        out[:, ok] = grid[:, rr[ok], cc[ok]]
        return out

    @staticmethod
    def _with_frontier(crop: np.ndarray) -> np.ndarray:
        known = (crop[C_FREE] > 0) | (crop[C_WALL] > 0) | (crop[C_DOOR] > 0) | (crop[C_LOCK] > 0)
        unknown = ~known
        nb = np.zeros_like(unknown)
        nb[1:, :] |= unknown[:-1, :]
        nb[:-1, :] |= unknown[1:, :]
        nb[:, 1:] |= unknown[:, :-1]
        nb[:, :-1] |= unknown[:, 1:]
        frontier = ((crop[C_FREE] > 0) & nb).astype(np.uint8) * 255
        return np.concatenate([crop, frontier[None]], 0)

    def crops(self, x: float, y: float, heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
        h = math.radians(heading_deg)
        fine = self._sample(self.grid, x, y, h, _FINE_OFF, FINE_RES)
        pooled = self.grid.reshape(N_CH, SIZE // 4, 4, SIZE // 4, 4).max(axis=(2, 4))
        coarse = self._sample(pooled, x, y, h, _COARSE_OFF, COARSE_RES)
        return self._with_frontier(fine), self._with_frontier(coarse)
