"""Honest privileged teacher (TRAINING ONLY; never imported by submission/).

Uses the true map grid and true pose (privileged) for path search and aiming, but only *knows* what the
agent has already observed: cells swept by its rays, exits/keys seen inside the 90° field of view,
enemies in `visible_entities`. So every teacher decision is learnable from the student's own history.

Priorities: FIGHT > PICKUP (low health) > GO_EXIT > GET_KEY > EXPLORE (nearest frontier); doors and
locked doors on the path are opened with a single USE press (USE fires on the button-down edge).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from training.mapgen.grid import DOOR, EXIT, FLOOR, SWITCH, WALL, MapSpec

INTENT = {"explore": 0, "fight": 1, "get_key": 2, "open_door": 3, "go_exit": 4, "pickup": 5}
LOCK_COLOR = {"B": 0, "Y": 1, "R": 2}          # vitals key order: blue, yellow, red (cards), then skulls
KEY_IDX = {"blue": 0, "yellow": 1, "red": 2}
CELL = 64.0
RAY_ANG = np.arange(64) * (2 * math.pi / 64)
SAMPLES = np.arange(16.0, 1024.0, 24.0)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
ENEMY_T, PROJ_T, KEY_T = (1, 2, 3), 4, 9
PICKUP_T = (5, 6, 7, 8)


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class Teacher:
    def __init__(self, spec: MapSpec, rng: np.random.Generator | None = None):
        self.spec = spec
        self.grid = spec.grid
        self.H, self.W = self.grid.shape
        self.rng = rng or np.random.default_rng(spec.seed)
        self.seen = np.zeros((self.H, self.W), bool)
        self.exit_known = False
        self.key_known = {color: False for color, _ in spec.keys}
        self.key_cells = {color: cell for color, cell in spec.keys}
        self.exit_targets = self._exit_targets()
        self.exit_mask = (self.grid == EXIT) | (self.grid == SWITCH)
        self.exit_in_view = False
        self.items_seen: dict[tuple[int, int], int] = {}
        self.prev_use = 0
        self.use_cooldown = 0
        self.stuck = 0
        self.unstick = 0
        self.unstick_action = None
        self.last_pos = None
        self.path: list[tuple[int, int]] = []
        self.target: tuple[int, int] | None = None
        self.intent = INTENT["explore"]
        self.replan = 0
        self.strafe_dir = 1
        self.strafe_t = 0

    # ------------------------------------------------------------------ knowledge
    def _exit_targets(self) -> list[tuple[int, int]]:
        out = []
        for r, c in zip(*np.where((self.grid == EXIT) | (self.grid == SWITCH))):
            out.append((int(r), int(c)))
        return out

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(y / CELL)), int(math.floor(x / CELL))

    def observe(self, obs: dict, st: dict) -> None:
        x, y, h = st["x"], st["y"], math.radians(st["angle"])
        self.exit_in_view = False
        dist = obs["rays"]["distance"] * 1024.0
        ang = h + RAY_ANG
        ca, sa = np.cos(ang), np.sin(ang)
        # all ray samples at once (vectorized)
        mask = SAMPLES[None, :] < dist[:, None]
        rr = np.floor((y + sa[:, None] * SAMPLES[None, :]) / CELL).astype(np.int64)
        cc = np.floor((x + ca[:, None] * SAMPLES[None, :]) / CELL).astype(np.int64)
        ok = mask & (rr >= 0) & (rr < self.H) & (cc >= 0) & (cc < self.W)
        self.seen[rr[ok], cc[ok]] = True
        hr = np.floor((y + sa * (dist + 4)) / CELL).astype(np.int64)
        hc = np.floor((x + ca * (dist + 4)) / CELL).astype(np.int64)
        hok = (hr >= 0) & (hr < self.H) & (hc >= 0) & (hc < self.W)
        self.seen[hr[hok], hc[hok]] = True
        # forward field of view (±45°): exit floors / switches are visible in pixels
        fov = np.zeros(64, bool)
        fov[:9] = True
        fov[56:] = True
        fov_ok = ok & fov[:, None]
        hit_fov = hok & fov
        if self.exit_mask[rr[fov_ok], cc[fov_ok]].any() or self.exit_mask[hr[hit_fov], hc[hit_fov]].any():
            self.exit_known = True
            self.exit_in_view = True
        # keys / items in sight
        ent = obs["visible_entities"]
        for k in np.flatnonzero(ent["mask"]):
            f = ent["features"][k]
            et = int(ent["entity_type"][k])
            ea = h + math.atan2(f[1], f[2])
            ex, ey = x + math.cos(ea) * f[0] * 1024.0, y + math.sin(ea) * f[0] * 1024.0
            cell = self._cell(ex, ey)
            if et == KEY_T:
                for color, kc in self.key_cells.items():
                    if abs(kc[0] - cell[0]) <= 1 and abs(kc[1] - cell[1]) <= 1:
                        self.key_known[color] = True
            elif et in PICKUP_T:
                self.items_seen[cell] = et

    # ------------------------------------------------------------------ navigation
    def _passable(self, cell, keys_held) -> bool:
        ch = self.grid[cell]
        if ch in (FLOOR, DOOR, EXIT):
            return True
        if ch in LOCK_COLOR:
            return bool(keys_held[LOCK_COLOR[ch]] or keys_held[LOCK_COLOR[ch] + 3])
        return False

    def _bfs(self, start, keys_held, goal_fn, through_unknown=False):
        """BFS over seen passable cells. Returns path to the first cell with goal_fn(cell) True."""
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur != start and goal_fn(cur):
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            for dr, dc in DIRS:
                n = (cur[0] + dr, cur[1] + dc)
                if not (0 <= n[0] < self.H and 0 <= n[1] < self.W) or n in prev:
                    continue
                if self.grid[n] == SWITCH and goal_fn(n):
                    prev[n] = cur
                    q.append(n)
                    continue
                if not self._passable(n, keys_held):
                    continue
                if not self.seen[n] and not through_unknown:
                    # unknown cells may be goals (frontier) but are not expanded
                    if goal_fn(n):
                        prev[n] = cur
                        q.append(n)
                    continue
                prev[n] = cur
                q.append(n)
        return None

    def _frontier_goal(self, cell) -> bool:
        if not self.seen[cell] or self.grid[cell] == WALL:
            return False
        for dr, dc in DIRS:
            n = (cell[0] + dr, cell[1] + dc)
            if 0 <= n[0] < self.H and 0 <= n[1] < self.W and not self.seen[n] and self.grid[n] != WALL:
                return True
        return False

    def _plan(self, me, keys_held, health: float) -> None:
        held = lambda color: bool(keys_held[KEY_IDX[color]] or keys_held[KEY_IDX[color] + 3])  # noqa: E731
        plans = []
        if health < 40 and self.items_seen:
            p = self._bfs(me, keys_held, lambda c: c in self.items_seen and self.items_seen[c] in (5, 6))
            if p and len(p) <= 12:
                plans.append(("pickup", p))
        if self.exit_known:
            p = self._bfs(me, keys_held, lambda c: c in self.exit_targets)
            if p:
                plans.append(("go_exit", p))
        for color, known in self.key_known.items():
            if known and not held(color):
                p = self._bfs(me, keys_held, lambda c, kc=self.key_cells[color]: c == kc)
                if p:
                    plans.append(("get_key", p))
        p = self._bfs(me, keys_held, self._frontier_goal)
        if p:
            plans.append(("explore", p))
        # nearby pickups are worth a short detour while exploring
        if self.items_seen:
            p = self._bfs(me, keys_held, lambda c: c in self.items_seen)
            if p and len(p) <= 4:
                plans.append(("pickup", p))
            weapons = self._bfs(me, keys_held, lambda c: c in self.items_seen and self.items_seen[c] == 8)
            if weapons and len(weapons) <= 10:
                plans.append(("pickup", weapons))
        if not plans:
            self.path, self.target, self.intent = [], None, INTENT["explore"]
            return
        order = {"pickup": 0, "go_exit": 1, "get_key": 2, "explore": 3}
        # low-health pickup first, then exit, then keys, then exploration; short pickups before exploring
        plans.sort(key=lambda kv: (order[kv[0]] if not (kv[0] == "pickup" and len(kv[1]) > 4 and health >= 40) else 9,
                                   len(kv[1])))
        name, path = plans[0]
        self.intent = INTENT[name]
        self.path = path
        self.target = path[-1]

    # ------------------------------------------------------------------ control
    def act(self, obs: dict, st: dict) -> tuple[np.ndarray, int]:
        """Return (9-button action, intent id) for the current observation and privileged state."""
        self.observe(obs, st)
        x, y, ang = st["x"], st["y"], st["angle"]
        me = self._cell(x, y)
        keys_held = obs["vitals"][8:14]
        health = float(obs["vitals"][0])
        for item_cell in list(self.items_seen):
            if abs(item_cell[0] - me[0]) + abs(item_cell[1] - me[1]) == 0:
                del self.items_seen[item_cell]
        a = np.zeros(9, np.int8)

        # stuck handling
        if self.last_pos is not None and math.hypot(x - self.last_pos[0], y - self.last_pos[1]) < 1.5:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_pos = (x, y)
        if self.unstick > 0:
            self.unstick -= 1
            self.prev_use = 0
            return self.unstick_action.copy(), self.intent
        if self.stuck >= 8:
            self.stuck = 0
            self.unstick = 4
            self.unstick_action = np.zeros(9, np.int8)
            self.unstick_action[int(self.rng.choice([1, 2, 3]))] = 1
            self.unstick_action[int(self.rng.choice([4, 5]))] = 1
            self.replan = 0
            return self.unstick_action.copy(), self.intent

        # FIGHT: nearest visible enemy
        ent = obs["visible_entities"]
        enemies = [k for k in np.flatnonzero(ent["mask"]) if int(ent["entity_type"][k]) in ENEMY_T
                   and ent["features"][k][0] * 1024 < 1100]
        if enemies:
            k = enemies[0]
            f = ent["features"][k]
            rel = math.degrees(math.atan2(f[1], f[2]))
            dist = f[0] * 1024.0
            tol = 4.0 if dist > 400 else 8.0
            if rel > tol:
                a[4] = 1
            elif rel < -tol:
                a[5] = 1
            if abs(rel) < max(tol * 1.5, 6.0):
                a[6] = 1
            # dodge: strafe, back off from melee enemies
            self.strafe_t -= 1
            if self.strafe_t <= 0:
                self.strafe_dir = -self.strafe_dir
                self.strafe_t = int(self.rng.integers(4, 10))
            a[2 if self.strafe_dir > 0 else 3] = 1
            # back off from melee enemies, and from everything when badly hurt or outnumbered
            n_close = sum(1 for j in enemies if ent["features"][j][0] * 1024 < 600)
            if (int(ent["entity_type"][k]) == 1 and dist < 250) or (health < 35 and dist < 700) or n_close >= 3:
                a[1] = 1
            if obs["vitals"][3] <= 0:
                a[8] = 1 if self.rng.random() < 0.3 else 0
            # dodge incoming projectiles harder
            proj = [j for j in np.flatnonzero(ent["mask"]) if int(ent["entity_type"][j]) == PROJ_T
                    and ent["features"][j][0] * 1024 < 350]
            if proj:
                a[2], a[3] = (1, 0) if self.strafe_dir > 0 else (0, 1)
            self.prev_use = 0
            return a, INTENT["fight"]

        # NAVIGATE
        self.replan -= 1
        if self.replan <= 0 or not self.path or me not in self.path:
            self._plan(me, keys_held, health)
            self.replan = 4
        if not self.path:
            a[0] = 1
            a[4] = 1
            self.prev_use = 0
            return a, INTENT["explore"]
        idx = self.path.index(me)
        nxt = self.path[min(idx + 1, len(self.path) - 1)]
        # look further ahead along straight segments
        look = nxt
        for j in range(idx + 1, min(idx + 4, len(self.path))):
            cand = self.path[j]
            if self._line_clear(me, cand):
                look = cand
            else:
                break
        tx, ty = look[1] * CELL + CELL / 2, look[0] * CELL + CELL / 2
        want = math.degrees(math.atan2(ty - y, tx - x))
        diff = wrap_deg(want - ang)
        if diff > 8:
            a[4] = 1
        elif diff < -8:
            a[5] = 1
        if abs(diff) < 50:
            a[0] = 1

        intent = self.intent
        # doors: next cell is a closed door / lock and we face it → press USE once, then keep pushing
        nch = self.grid[nxt]
        if nch == DOOR or nch in LOCK_COLOR or nch == SWITCH:
            ahead_type = int(obs["rays"]["hit_type"][0])
            ahead_d = float(obs["rays"]["distance"][0]) * 1024.0
            near = ahead_type in (1, 2, 3) and ahead_d < 90
            self.use_cooldown -= 1
            if (near or nch == SWITCH and ahead_d < 90) and abs(diff) < 30 and self.prev_use == 0 and self.use_cooldown <= 0:
                a[7] = 1
                self.use_cooldown = 16  # a door takes ~1 s to open; pressing again would reverse it
            if nch != SWITCH:
                intent = INTENT["open_door"]
        self.prev_use = int(a[7])
        return a, intent

    def _line_clear(self, a, b) -> bool:
        (r0, c0), (r1, c1) = a, b
        n = max(abs(r1 - r0), abs(c1 - c0)) * 3 + 1
        for t in np.linspace(0, 1, n):
            r = int(round(r0 + (r1 - r0) * t))
            c = int(round(c0 + (c1 - c0) * t))
            if self.grid[r, c] in (WALL, SWITCH) or self.grid[r, c] in LOCK_COLOR or self.grid[r, c] == DOOR:
                if (r, c) != b:
                    return False
        return True
