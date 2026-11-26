"""Difficulty tiers for procedural training maps (mirrors the 4 hidden evaluation tiers)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Doom thing type ids (shared by Doom 2 / Freedoom 2).
MONSTERS = {
    "zombieman": 3004, "shotgunguy": 9, "chaingunner": 65, "imp": 3001, "demon": 3002,
    "spectre": 58, "lostsoul": 3006, "cacodemon": 3005, "hellknight": 69,
}
ITEMS = {
    "stimpack": 2011, "medikit": 2012, "greenarmor": 2018, "bluearmor": 2019, "clip": 2007,
    "shells": 2008, "shotgun": 2001, "chaingun": 2002, "ammobox": 2048, "shellbox": 2049, "barrel": 2035,
}
KEY_CARDS = {"red": 13, "blue": 5, "yellow": 6}
KEY_SKULLS = {"red": 38, "blue": 40, "yellow": 39}
LOCK_NUMBER = {"red": 129, "blue": 130, "yellow": 131}  # ZDoom "any red/blue/yellow key" (card or skull)
LOCK_CHAR = {"red": "R", "blue": "B", "yellow": "Y"}
PLAYER_START = 1


@dataclass
class TierConfig:
    tier: int
    size: tuple[int, int]            # grid side length range (cells of 64 units)
    rooms: tuple[int, int]
    room_size: tuple[int, int]
    loop_prob: float                 # extra connections beyond the spanning tree
    door_prob: float                 # chance a 1-wide doorway gets a door
    locks: tuple[int, int]           # number of locked doors gating the exit
    monsters: tuple[int, int]
    monster_mix: dict[str, float]
    items: tuple[int, int]
    item_mix: dict[str, float]
    ambush_prob: float = 0.0         # monsters placed deaf in closed rooms
    exit_switch_prob: float = 0.15   # exit as a wall switch instead of a walk-over GATE floor
    skull_prob: float = 0.2          # skull key instead of keycard
    episode_ticks: tuple[int, int] = (2100, 4200)
    extra: dict = field(default_factory=dict)


EASY_MIX = {"zombieman": 4, "shotgunguy": 2, "imp": 3, "demon": 1}
MID_MIX = {"zombieman": 3, "shotgunguy": 3, "imp": 3, "demon": 2, "chaingunner": 1, "lostsoul": 1, "spectre": 0.5}
HARD_MIX = {"zombieman": 2, "shotgunguy": 3, "imp": 3, "demon": 2, "chaingunner": 2, "lostsoul": 1,
            "spectre": 1, "cacodemon": 1, "hellknight": 0.5}
ITEM_MIX = {"stimpack": 3, "medikit": 1.5, "clip": 3, "shells": 2, "greenarmor": 0.7, "shotgun": 1,
            "ammobox": 0.5, "shellbox": 0.5, "chaingun": 0.4, "barrel": 0.6, "bluearmor": 0.2}
SCARCE_ITEMS = {"stimpack": 2.5, "clip": 2, "shells": 1.5, "shotgun": 1.5, "medikit": 0.7, "greenarmor": 0.5, "barrel": 0.8}

TIERS: dict[int, TierConfig] = {
    1: TierConfig(1, (12, 20), (1, 3), (4, 9), 0.3, 0.2, (0, 0), (3, 9), EASY_MIX, (3, 7), ITEM_MIX,
                  episode_ticks=(2100, 2100)),
    2: TierConfig(2, (20, 34), (5, 10), (3, 8), 0.25, 0.45, (0, 0), (2, 7), MID_MIX, (3, 8), ITEM_MIX,
                  episode_ticks=(2800, 3500)),
    3: TierConfig(3, (22, 38), (6, 11), (3, 8), 0.2, 0.4, (1, 2), (3, 9), MID_MIX, (4, 9), ITEM_MIX,
                  episode_ticks=(3500, 4200)),
    4: TierConfig(4, (26, 44), (8, 14), (3, 8), 0.2, 0.45, (2, 3), (6, 14), HARD_MIX, (4, 8), SCARCE_ITEMS,
                  ambush_prob=0.4, episode_ticks=(4200, 4200)),
}
