"""Generate maps: uv run python -m training.mapgen.cli --tier 3 --seed 7 --out /tmp/yodha_maps [--show]"""
from __future__ import annotations

import argparse
from pathlib import Path

from .grid import generate
from .udmf import write_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--out", default="/tmp/yodha_maps")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    for i in range(a.count):
        spec = generate(a.tier, a.seed + i)
        path = write_map(spec, Path(a.out) / f"t{a.tier}_s{a.seed + i}.wad")
        print(f"{path}  size={spec.grid.shape} ticks={spec.episode_ticks} locks={[c for c, _ in spec.locks]} "
              f"things={len(spec.things)}")
        if a.show:
            print(spec.ascii())


if __name__ == "__main__":
    main()
