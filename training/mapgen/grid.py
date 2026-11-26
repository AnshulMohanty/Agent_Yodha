"""Procedural grid layouts: rooms + corridors + doors + key/lock progression + monsters/items.

Grid characters (structure only; things live in `MapSpec.things`):
  '#' wall   '.' floor   'D' door   'R'/'B'/'Y' locked door (red/blue/yellow)
  'E' exit floor cell (walk-over exit line)   'X' exit switch (a wall cell whose faces are switches)

Cell (r, c) covers x in [64c, 64c+64], y in [64r, 64r+64] (the organizers' fixture convention).
Training-only code: BFS and planning here never reach `submission/`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .tiers import ITEMS, KEY_CARDS, KEY_SKULLS, LOCK_CHAR, MONSTERS, PLAYER_START, TIERS, TierConfig

WALL, FLOOR, DOOR, EXIT, SWITCH = "#", ".", "D", "E", "X"
LOCKS = set(LOCK_CHAR.values())
PASSABLE = {FLOOR, DOOR, EXIT} | LOCKS
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass
class Thing:
    r: int
    c: int
    type: int
    angle: int = 0
    kind: str = ""        # "player" | "monster" | "item" | "key"
    name: str = ""
    ambush: bool = False


@dataclass
class MapSpec:
    grid: np.ndarray                      # (H, W) array of single characters
    things: list[Thing]
    tier: int
    seed: int
    episode_ticks: int
    start: tuple[int, int]
    exit_cells: list[tuple[int, int]]
    locks: list[tuple[str, tuple[int, int]]] = field(default_factory=list)   # (color, cell)
    keys: list[tuple[str, tuple[int, int]]] = field(default_factory=list)    # (color, cell)
    rooms: list[tuple[int, int, int, int]] = field(default_factory=list)     # (r0, c0, h, w)

    def ascii(self) -> str:
        g = self.grid.copy()
        sym = {"player": "P", "monster": "m", "item": "i", "key": "k"}
        for t in self.things:
            if g[t.r, t.c] == FLOOR:
                g[t.r, t.c] = sym.get(t.kind, "?") if t.kind != "key" else t.name[0]
        return "\n".join("".join(row) for row in g[::-1])  # print with +y up


class GenerationError(RuntimeError):
    pass


def _bfs(grid: np.ndarray, start: tuple[int, int], passable) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist and passable(grid[nr, nc]):
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def _path(grid, start, goal, passable) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            out = []
            while cur is not None:
                out.append(cur)
                cur = prev[cur]
            return out[::-1]
        r, c = cur
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in prev and passable(grid[nr, nc]):
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    return None


def _weighted_choice(rng, mix: dict[str, float]) -> str:
    names = list(mix)
    p = np.array([mix[n] for n in names], float)
    return names[rng.choice(len(names), p=p / p.sum())]


def _carve_rooms(rng, cfg: TierConfig, H: int, W: int, grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    target = int(rng.integers(cfg.rooms[0], cfg.rooms[1] + 1))
    for _ in range(400):
        if len(rooms) >= target:
            break
        h = int(rng.integers(cfg.room_size[0], cfg.room_size[1] + 1))
        w = int(rng.integers(cfg.room_size[0], cfg.room_size[1] + 1))
        if h + 2 > H or w + 2 > W:
            continue
        r0 = int(rng.integers(1, H - h))
        c0 = int(rng.integers(1, W - w))
        ok = all(r0 + h + 1 <= rr0 or rr0 + rh + 1 <= r0 or c0 + w + 1 <= rc0 or rc0 + rw + 1 <= c0
                 for rr0, rc0, rh, rw in rooms)
        if ok:
            rooms.append((r0, c0, h, w))
            grid[r0:r0 + h, c0:c0 + w] = FLOOR
    if not rooms:
        raise GenerationError("no rooms placed")
    return rooms


def _carve_corridor(rng, grid, a: tuple[int, int], b: tuple[int, int], width: int) -> None:
    (r1, c1), (r2, c2) = a, b
    H, W = grid.shape

    def carve(r, c):
        for dr in range(width):
            for dc in range(width):
                rr, cc = r + dr, c + dc
                if 1 <= rr < H - 1 and 1 <= cc < W - 1:
                    grid[rr, cc] = FLOOR

    if rng.random() < 0.5:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            carve(r1, c)
        for r in range(min(r1, r2), max(r1, r2) + 1):
            carve(r, c2)
    else:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            carve(r, c1)
        for c in range(min(c1, c2), max(c1, c2) + 1):
            carve(r2, c)


def _connect(rng, cfg: TierConfig, grid, rooms) -> None:
    centers = [(r0 + int(rng.integers(0, h)), c0 + int(rng.integers(0, w))) for r0, c0, h, w in rooms]
    n = len(centers)
    if n < 2:
        return
    # Prim's MST on Manhattan distance, plus random extra edges for loops.
    in_tree, edges = {0}, []
    while len(in_tree) < n:
        best = None
        for i in in_tree:
            for j in range(n):
                if j in in_tree:
                    continue
                d = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        in_tree.add(j)
        edges.append((i, j))
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges and (j, i) not in edges and rng.random() < cfg.loop_prob / n:
                edges.append((i, j))
    for i, j in edges:
        width = 2 if rng.random() < 0.2 else 1
        _carve_corridor(rng, grid, centers[i], centers[j], width)


def _doorway_cells(grid) -> list[tuple[int, int]]:
    """1-wide passage cells: walls on two opposite sides, floor on the other two."""
    H, W = grid.shape
    out = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if grid[r, c] != FLOOR:
                continue
            ns_wall = grid[r - 1, c] == WALL and grid[r + 1, c] == WALL
            ew_wall = grid[r, c - 1] == WALL and grid[r, c + 1] == WALL
            ns_floor = grid[r - 1, c] != WALL and grid[r + 1, c] != WALL
            ew_floor = grid[r, c - 1] != WALL and grid[r, c + 1] != WALL
            if (ns_wall and ew_floor) or (ew_wall and ns_floor):
                out.append((r, c))
    return out


def _room_of(rooms, cell) -> int:
    r, c = cell
    for i, (r0, c0, h, w) in enumerate(rooms):
        if r0 <= r < r0 + h and c0 <= c < c0 + w:
            return i
    return -1


def _place_exit(rng, grid, rooms, start) -> tuple[list[tuple[int, int]], str]:
    """Exit alcove (a wall cell next to exactly one floor cell) in the room farthest from the start."""
    dist = _bfs(grid, start, lambda ch: ch in PASSABLE)
    room_d = []
    for i, (r0, c0, h, w) in enumerate(rooms):
        cells = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w) if (r, c) in dist]
        if cells:
            room_d.append((max(dist[x] for x in cells), i))
    room_d.sort(reverse=True)
    H, W = grid.shape
    for _, i in room_d[:3]:
        r0, c0, h, w = rooms[i]
        cand = []
        for r in range(r0 - 1, r0 + h + 1):
            for c in range(c0 - 1, c0 + w + 1):
                if not (1 <= r < H - 1 and 1 <= c < W - 1) or grid[r, c] != WALL:
                    continue
                nbrs = [(r + dr, c + dc) for dr, dc in DIRS]
                floors = [n for n in nbrs if grid[n] in (FLOOR,)]
                if len(floors) == 1 and _room_of(rooms, floors[0]) == i:
                    cand.append(((r, c), floors[0]))
        if cand:
            (cell, _), = [cand[int(rng.integers(len(cand)))]]
            return [cell], "alcove"
    # Fallback: farthest reachable floor cell becomes the exit.
    far = max(dist, key=dist.get)
    return [far], "floor"


def _articulation_on_path(grid, start, goal, passable) -> list[tuple[int, int]]:
    path = _path(grid, start, goal, passable)
    if path is None:
        return []
    cands = []
    for cell in path[2:-2]:
        if grid[cell] != FLOOR:
            continue
        saved = grid[cell]
        grid[cell] = WALL
        blocked = goal not in _bfs(grid, start, passable)
        grid[cell] = saved
        if blocked:
            cands.append(cell)
    return cands


def generate(tier: int, seed: int, cfg: TierConfig | None = None, max_tries: int = 50) -> MapSpec:
    cfg = cfg or TIERS[tier]
    for attempt in range(max_tries):
        rng = np.random.default_rng([seed, attempt])
        try:
            spec = _generate_once(rng, cfg, seed)
        except GenerationError:
            continue
        from .solvable import check

        if check(spec).solvable:
            return spec
    raise GenerationError(f"could not generate a solvable tier-{tier} map for seed {seed}")


def _generate_once(rng, cfg: TierConfig, seed: int) -> MapSpec:
    H = int(rng.integers(cfg.size[0], cfg.size[1] + 1))
    W = int(rng.integers(cfg.size[0], cfg.size[1] + 1))
    grid = np.full((H, W), WALL, dtype="<U1")
    rooms = _carve_rooms(rng, cfg, H, W, grid)
    _connect(rng, cfg, grid, rooms)

    r0, c0, h, w = rooms[0]
    start = (r0 + int(rng.integers(0, h)), c0 + int(rng.integers(0, w)))

    exit_cells, exit_kind = _place_exit(rng, grid, rooms, start)
    use_switch = exit_kind == "alcove" and rng.random() < cfg.exit_switch_prob
    for cell in exit_cells:
        grid[cell] = SWITCH if use_switch else EXIT
    if start in exit_cells:
        raise GenerationError("start on exit")

    # Doors on 1-wide doorways (not next to the exit alcove or the start).
    for cell in _doorway_cells(grid):
        if cell == start or any(abs(cell[0] - e[0]) + abs(cell[1] - e[1]) <= 1 for e in exit_cells):
            continue
        if rng.random() < cfg.door_prob:
            grid[cell] = DOOR

    # Key/lock progression: each lock gates the previous objective (exit first, then the previous key).
    colors = list(rng.permutation(["red", "blue", "yellow"]))
    n_locks = int(rng.integers(cfg.locks[0], cfg.locks[1] + 1))
    passable_now = lambda ch: ch in PASSABLE and ch not in LOCKS  # noqa: E731
    exit_goal = exit_cells[0] if not use_switch else _switch_access(grid, exit_cells[0])
    locks, keys = [], []
    target = exit_goal
    for i in range(n_locks):
        cands = _articulation_on_path(grid, start, target, passable_now)
        if not cands:
            break
        # Keep only gates that leave a big explorable region (and so a far-away key) before them.
        total = len(_bfs(grid, start, passable_now))
        sized = []
        for cell in cands:
            grid[cell] = WALL
            pre = len(_bfs(grid, start, passable_now))
            grid[cell] = FLOOR
            sized.append((pre, cell))
        good = [cell for pre, cell in sized if pre >= 0.3 * total and pre >= 25
                and not any(grid[cell[0] + dr, cell[1] + dc] in LOCKS for dr, dc in DIRS)]
        if not good:
            if i == 0:
                raise GenerationError("no good gate")
            break
        gate = good[int(rng.integers(len(good)))]
        color = colors[i]
        grid[gate] = LOCK_CHAR[color]
        reach = _bfs(grid, start, passable_now)
        reach.pop(start, None)
        # Prefer far-away cells off the main path so the agent must explore a branch and come back.
        main = set(_path(grid, start, gate, passable_now) or [])
        pool = [(d, cell) for cell, d in reach.items() if grid[cell] == FLOOR and cell not in main]
        if not pool:
            pool = [(d, cell) for cell, d in reach.items() if grid[cell] == FLOOR]
        if not pool:
            raise GenerationError("no key cell")
        pool.sort()
        key_cell = pool[int(rng.integers(len(pool) // 2, len(pool)))][1]
        locks.append((color, gate))
        keys.append((color, key_cell))
        target = key_cell
    if cfg.locks[0] > 0 and not locks:
        raise GenerationError("tier needs locks but no gate found")
    _separate_doors(grid)

    things: list[Thing] = [Thing(start[0], start[1], PLAYER_START, int(rng.integers(0, 8)) * 45, "player", "player")]
    occupied = {start} | {k for _, k in keys}
    for color, cell in keys:
        skull = rng.random() < cfg.skull_prob
        things.append(Thing(cell[0], cell[1], (KEY_SKULLS if skull else KEY_CARDS)[color], 0, "key", color))

    dist = _bfs(grid, start, lambda ch: ch in PASSABLE)
    floor_cells = [cell for cell, d in dist.items() if grid[cell] == FLOOR and cell not in occupied]
    rng.shuffle(floor_cells)

    n_mon = int(rng.integers(cfg.monsters[0], cfg.monsters[1] + 1))
    far_cells = [c for c in floor_cells if dist[c] >= 5]
    for cell in far_cells[:n_mon]:
        name = _weighted_choice(rng, cfg.monster_mix)
        ambush = rng.random() < cfg.ambush_prob
        things.append(Thing(cell[0], cell[1], MONSTERS[name], int(rng.integers(0, 8)) * 45, "monster", name, ambush))
        occupied.add(cell)

    n_items = int(rng.integers(cfg.items[0], cfg.items[1] + 1))
    free = [c for c in floor_cells if c not in occupied]
    for cell in free[:n_items]:
        name = _weighted_choice(rng, cfg.item_mix)
        things.append(Thing(cell[0], cell[1], ITEMS[name], 0, "item", name))

    ticks = int(rng.integers(cfg.episode_ticks[0], cfg.episode_ticks[1] + 1))
    ticks = int(round(ticks / 4) * 4)
    return MapSpec(grid, things, cfg.tier, seed, ticks, start, exit_cells, locks, keys, rooms)


def _separate_doors(grid) -> None:
    """A Doom door opens to its lowest neighbouring ceiling, so a door touching another closed door/lock
    sector could never open. Turn plain doors that touch a different door/lock kind back into floor."""
    H, W = grid.shape
    changed = True
    while changed:
        changed = False
        for r in range(H):
            for c in range(W):
                ch = grid[r, c]
                if ch != DOOR:
                    continue
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] in LOCKS:
                        grid[r, c] = FLOOR
                        changed = True
                        break


def _switch_access(grid, switch_cell) -> tuple[int, int]:
    for dr, dc in DIRS:
        n = (switch_cell[0] + dr, switch_cell[1] + dc)
        if grid[n] == FLOOR:
            return n
    raise GenerationError("switch has no floor neighbour")
