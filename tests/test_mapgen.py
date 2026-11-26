"""Our procedural generator: maps load in the official wrapper, are solvable, and doors/locks/exits work."""
import warnings

import numpy as np
import pytest

from doom_env import make_env
from doom_env import spaces as S
from training.mapgen.grid import MapSpec, Thing, generate
from training.mapgen.solvable import check
from training.mapgen.tiers import KEY_CARDS, PLAYER_START
from training.mapgen.udmf import write_map

warnings.filterwarnings("ignore")
FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)


def corridor_spec(layout: str, things: list[Thing]) -> MapSpec:
    rows = layout.splitlines()
    grid = np.array([list(r) for r in rows], dtype="<U1")
    exits = [tuple(x) for x in np.argwhere(grid == "E")]
    start = next((t.r, t.c) for t in things if t.kind == "player")
    keys = [(t.name, (t.r, t.c)) for t in things if t.kind == "key"]
    return MapSpec(grid, things, 3, 0, 1000, start, exits, [], keys, [])


def test_key_lock_exit_roundtrip(tmp_path):
    spec = corridor_spec("############\n#..........E\n############".replace("E", "#")[:-1] + "#",
                         [Thing(1, 1, PLAYER_START, 0, "player")])
    # Build: P . r . . R . . . . E  (key red at col 3, red lock at col 6, exit at col 10)
    spec.grid[1, 6] = "R"
    spec.grid[1, 10] = "E"
    spec.exit_cells = [(1, 10)]
    spec.things.append(Thing(1, 3, KEY_CARDS["red"], 0, "key", "red"))
    spec.keys = [("red", (1, 3))]
    assert check(spec).solvable
    env = make_env(str(write_map(spec, tmp_path / "m.wad")), "MAP01", episode_ticks=1000)
    obs, _ = env.reset(seed=0)
    k = 0  # ray straight ahead
    assert obs["rays"]["hit_type"][k] == 5  # the key item is first in front
    picked, last_x = False, None
    for i in range(300):
        st = env.privileged_state()
        stuck = last_x is not None and abs(st["x"] - last_x) < 0.5
        last_x = st["x"]
        obs, r, term, trunc, info = env.step(USE if (stuck and i % 6 == 0) else FWD)
        picked |= obs["vitals"][S.VITALS_INDEX["key_red_card"]] == 1
        if term or trunc:
            break
    assert picked and term and info["exit_reached"]
    env.close()


@pytest.mark.parametrize("tier", [1, 2, 3, 4])
def test_generated_maps_load_and_solve(tmp_path, tier):
    for seed in range(5):
        spec = generate(tier, 500 + seed)
        assert check(spec).solvable
        env = make_env(str(write_map(spec, tmp_path / f"t{tier}_{seed}.wad")), "MAP01", episode_ticks=200)
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        if tier >= 3:
            assert any(v > 0 for v in env.mapinfo.door_sectors.values()), "locked door not detected"
        env.close()
