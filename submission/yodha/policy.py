"""Stateful inference wrapper around YodhaNet (used by submission/agent.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .features import AgentMemory
from .model import HEADS, YodhaNet, factored_to_buttons


def to_batch(inp: dict, device="cpu") -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).to(device) for k, v in inp.items()}


class YodhaPolicy:
    def __init__(self, weights_dir: str | Path, device: str = "cpu"):
        weights_dir = Path(weights_dir)
        cfg_path = weights_dir / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.temperature = float(cfg.get("temperature", 0.0))
        self.device = device
        self.net = YodhaNet(hidden=int(cfg.get("hidden", 512)), use_pixels=bool(cfg.get("use_pixels", True)))
        wpath = weights_dir / "yodha.pt"
        if wpath.exists():
            sd = torch.load(wpath, map_location="cpu", weights_only=True)
            self.net.load_state_dict(sd, strict=False)
        self.net.to(device).eval()
        odo = weights_dir / "odometry.json"
        self.memory = AgentMemory(odo if odo.exists() else None)
        self.rng = np.random.default_rng(0)
        self.h = self.net.initial_state(1, device)

    def reset(self) -> None:
        self.memory.reset()
        self.h = self.net.initial_state(1, self.device)
        self.rng = np.random.default_rng(0)

    @torch.inference_mode()
    def act(self, obs: dict) -> np.ndarray:
        inp = self.memory.observe(obs)
        self.h, intent_logits, logits = self.net.forward_step(to_batch(inp, self.device), self.h)
        self.last = {"inp": inp, "intent": torch.softmax(intent_logits[0].float(), -1).cpu().numpy(),
                     "heads": [torch.softmax(lg[0].float(), -1).cpu().numpy() for lg in logits]}
        idx = []
        for lg in logits:
            p = lg[0].float()
            if self.temperature > 0:
                probs = torch.softmax(p / self.temperature, -1).cpu().numpy().astype(np.float64)
                probs /= probs.sum()
                idx.append(int(self.rng.choice(len(probs), p=probs)))
            else:
                idx.append(int(torch.argmax(p)))
        action = factored_to_buttons(torch.tensor(idx)).numpy().astype(np.int8)
        self.memory.record_action(action)
        return action


assert len(HEADS) == 6
