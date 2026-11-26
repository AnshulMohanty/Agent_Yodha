"""Evaluation entry point: the YODHA agent.

Every action comes from the trained YodhaNet (submission/yodha/model.py). The agent keeps its own
memory (odometry + EgoMap built only from observations) and a GRU state; both are cleared in reset().
Contract: docs/CONTRACT.md §6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

import torch  # noqa: E402

from yodha.policy import YodhaPolicy  # noqa: E402


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once: load weights and warm up (one CPU thread, as in the evaluation sandbox)."""
        self.observation_space = observation_space
        self.action_space = action_space
        torch.set_num_threads(1)
        self.policy = YodhaPolicy(WEIGHTS_DIR, device="cpu")
        try:  # warm-up forward pass so the first act() of a map is fast
            self.policy.act(observation_space.sample())
        except Exception:
            pass
        self.policy.reset()
        self._zeros = np.zeros(action_space.shape, np.int8)

    def reset(self) -> None:
        """Called before every map: clear the GRU state, EgoMap, odometry and history."""
        self.policy.reset()

    def act(self, observation: dict) -> np.ndarray:
        try:
            return self.policy.act(observation)
        except Exception:  # safety net: never crash the evaluation (standing still keeps earned points)
            return self._zeros
