"""Grid → UDMF (namespace "zdoom") WAD, matching the organizers' fixture format.

- 64-unit cells; floor regions, door cells, locked-door cells and exit cells become sectors.
- Doors: own sector (ceiling 0), lines special 12 Door_Raise(0, 16, 150), room on the front side.
- Locked doors: special 13 Door_LockedRaise(0, 16, 0, lock), lock 129/130/131, DOORRED/BLU/YEL trim.
- Exit: GATE floor cell entered through lines with special 243 Exit_Normal (playercross),
  or an exit switch (special 243, playeruse, SW1EXIT) on the faces of an 'X' wall cell.
- Collinear edges with identical attributes are merged (fewer lines = faster sensors). No T-junctions
  can appear because a run only breaks where one of its two sides changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from doom_env.wad import Lump, write_wad

from .grid import DIRS, DOOR, EXIT, FLOOR, LOCKS, SWITCH, WALL, MapSpec
from .tiers import LOCK_NUMBER

CELL = 64
WALL_TEX = ["STARTAN2", "STARTAN3", "STARGR1", "BROWN1", "BROWNGRN", "GRAY1", "GRAY4", "STONE2", "STONE3",
            "BRICK1", "BRICK7", "CEMENT1", "METAL", "TEKGREN2", "BIGBRIK1", "ICKWALL3", "COMPBLUE", "BROWN96"]
FLOOR_TEX = ["FLOOR0_1", "FLOOR0_3", "FLOOR4_8", "FLOOR5_2", "FLAT1", "FLAT14", "FLAT5", "MFLR8_1", "FLOOR7_1"]
CEIL_TEX = ["CEIL1_1", "CEIL3_5", "CEIL5_1", "FLAT1", "CEIL3_3"]
DOOR_TEX = ["BIGDOOR2", "BIGDOOR1", "BIGDOOR4", "DOOR3"]
LOCK_TRIM = {"R": "DOORRED", "B": "DOORBLU", "Y": "DOORYEL"}
LOCK_COLOR = {"R": "red", "B": "blue", "Y": "yellow"}
GATE_TEX = ["GATE3", "GATE3", "GATE3", "GATE1", "GATE2", "GATE4"]


def _kind(ch: str) -> str:
    if ch in (WALL, SWITCH):
        return "wall"
    if ch == DOOR:
        return "door"
    if ch in LOCKS:
        return "lock" + ch
    if ch == EXIT:
        return "exit"
    return "floor"


def _sectors(grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Flood-fill same-kind cells into sectors. Returns (sector id grid, kind per sector)."""
    H, W = grid.shape
    sid = np.full((H, W), -1, np.int64)
    kinds: list[str] = []
    for r in range(H):
        for c in range(W):
            k = _kind(grid[r, c])
            if k == "wall" or sid[r, c] >= 0:
                continue
            n = len(kinds)
            kinds.append(k)
            stack = [(r, c)]
            sid[r, c] = n
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and sid[nr, nc] < 0 and _kind(grid[nr, nc]) == k:
                        sid[nr, nc] = n
                        stack.append((nr, nc))
    return sid, kinds


def build_textmap(spec: MapSpec, rng: np.random.Generator | None = None) -> str:
    rng = rng or np.random.default_rng(spec.seed)
    grid = spec.grid
    H, W = grid.shape
    sid, kinds = _sectors(grid)

    # ---- sector styles -------------------------------------------------------------------
    gate = str(rng.choice(GATE_TEX))
    door_tex = str(rng.choice(DOOR_TEX))
    sec_style = []
    for k in kinds:
        if k == "floor":
            ceil_h = int(rng.choice([104, 128, 128, 144, 160]))
            sec_style.append(dict(floor=str(rng.choice(FLOOR_TEX)), ceil=str(rng.choice(CEIL_TEX)), fh=0, ch=ceil_h,
                                  light=int(rng.integers(128, 256)), wall=str(rng.choice(WALL_TEX))))
        elif k == "exit":
            sec_style.append(dict(floor=gate, ceil="CEIL1_1", fh=0, ch=128, light=int(rng.integers(160, 256)),
                                  wall=str(rng.choice(WALL_TEX))))
        else:  # door / locked door: closed (ceiling = floor)
            sec_style.append(dict(floor="FLAT5_4", ceil="CEIL1_1", fh=0, ch=0, light=int(rng.integers(128, 256)),
                                  wall="DOORTRAK"))

    # ---- unit edges ------------------------------------------------------------------------
    # key: (axis, coord, front_positive, attrs) -> list of start offsets along the line
    runs: dict[tuple, list[int]] = {}

    def add_edge(axis: str, coord: int, start: int, front_pos: bool, attrs: tuple) -> None:
        runs.setdefault((axis, coord, front_pos, attrs), []).append(start)

    def cell_kind(r, c):
        if 0 <= r < H and 0 <= c < W:
            return grid[r, c]
        return WALL

    for r in range(-1, H):
        for c in range(-1, W):
            # vertical edge between (r, c) [left] and (r, c+1) [right] at x = 64(c+1)
            if 0 <= r < H:
                _edge(add_edge, sid, sec_style, cell_kind(r, c), cell_kind(r, c + 1),
                      sid[r, c] if 0 <= c < W else -1, sid[r, c + 1] if c + 1 < W else -1,
                      "v", CELL * (c + 1), CELL * r, door_tex)
            # horizontal edge between (r, c) [below] and (r+1, c) [above] at y = 64(r+1)
            if 0 <= c < W:
                _edge(add_edge, sid, sec_style, cell_kind(r, c), cell_kind(r + 1, c),
                      sid[r, c] if 0 <= r < H else -1, sid[r + 1, c] if r + 1 < H else -1,
                      "h", CELL * (r + 1), CELL * c, door_tex)

    # ---- merge runs and emit -----------------------------------------------------------------
    verts: dict[tuple[float, float], int] = {}
    vlist: list[tuple[float, float]] = []

    def vid(p):
        if p not in verts:
            verts[p] = len(vlist)
            vlist.append(p)
        return verts[p]

    sides: list[str] = []
    lines: list[str] = []
    for (axis, coord, front_pos, attrs), starts in sorted(runs.items(), key=lambda kv: str(kv[0])):
        starts.sort()
        segs, a = [], starts[0]
        prev = a
        for s in starts[1:]:
            if s != prev + CELL:
                segs.append((a, prev + CELL))
                a = s
            prev = s
        segs.append((a, prev + CELL))
        front_sec, back_sec, special, ftex, btex = attrs
        for lo, hi in segs:
            if axis == "v":  # x = coord, y in [lo, hi]
                p1, p2 = ((coord, lo), (coord, hi)) if front_pos else ((coord, hi), (coord, lo))
            else:            # y = coord, x in [lo, hi]
                p1, p2 = ((hi, coord), (lo, coord)) if front_pos else ((lo, coord), (hi, coord))
            v1, v2 = vid((float(p1[0]), float(p1[1]))), vid((float(p2[0]), float(p2[1])))
            fs = len(sides)
            sides.append(_sidedef(front_sec, ftex))
            if back_sec >= 0:
                bs = len(sides)
                sides.append(_sidedef(back_sec, btex))
                lines.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {fs}; sideback = {bs}; twosided = true;{special} }}")
            else:
                lines.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {fs}; blocking = true; dontpegbottom = true;{special} }}")

    out = ['namespace = "zdoom";']
    out += [f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}" for x, y in vlist]
    for st in sec_style:
        out.append(f'sector {{ texturefloor = "{st["floor"]}"; textureceiling = "{st["ceil"]}"; heightfloor = {st["fh"]}; '
                   f'heightceiling = {st["ch"]}; lightlevel = {st["light"]}; }}')
    out += sides
    out += lines
    for t in spec.things:
        x, y = CELL * t.c + CELL // 2, CELL * t.r + CELL // 2
        amb = " ambush = true;" if t.ambush else ""
        out.append(f"thing {{ x = {x:.1f}; y = {y:.1f}; angle = {t.angle}; type = {t.type}; skill1 = true; "
                   f"skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true;{amb} }}")
    return "\n".join(out) + "\n"


def _sidedef(sector: int, tex: tuple[str, str]) -> str:
    top, mid = tex
    return f'sidedef {{ sector = {sector}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "-"; }}'


def _edge(add, sid, styles, a_ch, b_ch, a_sec, b_sec, axis, coord, start, door_tex) -> None:
    """Edge between cell a (negative side) and cell b (positive side)."""
    ka, kb = _kind(a_ch), _kind(b_ch)
    if ka == "wall" and kb == "wall":
        return
    if ka != "wall" and kb != "wall" and a_sec == b_sec:
        return
    if ka == "wall" or kb == "wall":
        # one-sided line, front faces the open cell
        open_pos = ka == "wall"  # open cell is on the positive side
        sec = b_sec if open_pos else a_sec
        wall_ch = a_ch if open_pos else b_ch
        open_kind = kb if open_pos else ka
        special = ""
        tex = styles[sec]["wall"]
        if wall_ch == SWITCH and open_kind == "floor":
            special = " special = 243; playeruse = true;"
            tex = "SW1EXIT"
        add(axis, coord, start, open_pos, (sec, -1, special, ("-", tex), ("-", "-")))
        return
    # two-sided line between different sectors
    special = ""
    pair = {ka, kb}
    if "floor" in pair and ("door" in pair or any(k.startswith("lock") for k in pair) or "exit" in pair):
        front_pos = kb == "floor"  # the room is the front side
        other = ka if front_pos else kb
        front_sec, back_sec = (b_sec, a_sec) if front_pos else (a_sec, b_sec)
        if other == "door":
            special = " special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true;"
            ftex = (door_tex, "-")
        elif other.startswith("lock"):
            ch = other[-1]
            special = (f" special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {LOCK_NUMBER[LOCK_COLOR[ch]]}; "
                       "playeruse = true; monsteruse = false; repeatspecial = true;")
            ftex = (LOCK_TRIM[ch], "-")
        else:  # exit
            special = " special = 243; playercross = true;"
            ftex = ("-", "-")
        add(axis, coord, start, front_pos, (front_sec, back_sec, special, ftex, ("-", "-")))
        return
    # any other pair (door-door of different colors etc.): plain two-sided line
    add(axis, coord, start, True, (b_sec, a_sec, "", ("-", "-"), ("-", "-")))


def write_map(spec: MapSpec, path: str | Path, rng: np.random.Generator | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build_textmap(spec, rng)
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode("ascii")), Lump("ENDMAP", b"")])
    return path
