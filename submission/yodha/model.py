"""YodhaNet: recurrent multi-modal policy.

Encoders: circular ray conv, entity set-attention (any number of enemies), vitals/history MLP, small
pixel CNN (sees exits/doors/keys the rays cannot), two-scale EgoMap CNN. Core: GRU.
Heads: intent (hierarchical, FiLM-conditions the action heads), factorized action heads, value and
auxiliary heads (training only).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .egomap import CROP_CH
from .features import ENT_FEAT, NUM_ENT_TYPES, SCALAR_DIM

# factorized action heads: (name, number of choices)
HEADS = (("move", 3), ("strafe", 3), ("turn", 3), ("attack", 2), ("use", 2), ("weapon", 2))
INTENTS = ("explore", "fight", "get_key", "open_door", "go_exit", "pickup")
PRIV_DIM = 16  # privileged critic features (training only)
AUX = {"exit_visible": 1, "door_ahead": 1, "frontier_dir": 8}


def factored_to_buttons(idx: torch.Tensor) -> torch.Tensor:
    """(..., 6) head indices → (..., 9) MultiBinary buttons."""
    move, strafe, turn, attack, use, weapon = idx.unbind(-1)
    out = torch.zeros(idx.shape[:-1] + (9,), dtype=torch.int8, device=idx.device)
    out[..., 0] = (move == 1).to(torch.int8)
    out[..., 1] = (move == 2).to(torch.int8)
    out[..., 2] = (strafe == 1).to(torch.int8)
    out[..., 3] = (strafe == 2).to(torch.int8)
    out[..., 4] = (turn == 1).to(torch.int8)
    out[..., 5] = (turn == 2).to(torch.int8)
    out[..., 6] = attack.to(torch.int8)
    out[..., 7] = use.to(torch.int8)
    out[..., 8] = weapon.to(torch.int8)
    return out


def buttons_to_factored(b: torch.Tensor) -> torch.Tensor:
    """(..., 9) buttons → (..., 6) head indices (inverse of factored_to_buttons)."""
    b = b.long()
    move = torch.where(b[..., 0] == 1, 1, torch.where(b[..., 1] == 1, 2, 0))
    strafe = torch.where(b[..., 2] == 1, 1, torch.where(b[..., 3] == 1, 2, 0))
    turn = torch.where(b[..., 4] == 1, 1, torch.where(b[..., 5] == 1, 2, 0))
    return torch.stack([move, strafe, turn, b[..., 6], b[..., 7], b[..., 8]], -1)


class MapEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int, n: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * (n // 8) * (n // 8), out_dim)

    def forward(self, x):
        return F.relu(self.fc(self.conv(x).flatten(1)))


class YodhaNet(nn.Module):
    def __init__(self, hidden: int = 512, use_pixels: bool = True):
        super().__init__()
        self.hidden = hidden
        self.use_pixels = use_pixels
        # rays: (B, 64, 8) → circular conv over the ray index
        self.ray_conv = nn.ModuleList([
            nn.Conv1d(8, 32, 5, padding=2, padding_mode="circular"),
            nn.Conv1d(32, 64, 5, padding=2, padding_mode="circular"),
            nn.Conv1d(64, 64, 3, padding=1, padding_mode="circular"),
        ])
        self.ray_fc = nn.Linear(64 * 16, 256)
        # entities: set attention
        self.ent_type = nn.Embedding(NUM_ENT_TYPES, 16)
        self.ent_mlp = nn.Sequential(nn.Linear(ENT_FEAT + 16, 64), nn.ReLU(), nn.Linear(64, 64))
        self.ent_query = nn.Parameter(torch.randn(2, 64) * 0.1)  # "threat" and "goal" queries
        self.ent_attn = nn.MultiheadAttention(64, 4, batch_first=True)
        # scalars
        self.scalar_mlp = nn.Sequential(nn.Linear(SCALAR_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        # pixels (4, 60, 80)
        if use_pixels:
            self.pix = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4), nn.ReLU(),   # 14 x 19
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),  # 6 x 8
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(),  # 4 x 6
                nn.Flatten(), nn.Linear(64 * 4 * 6, 256), nn.ReLU(),
            )
        self.map_fine = MapEncoder(CROP_CH, 256, 64)
        self.map_coarse = MapEncoder(CROP_CH, 128, 32)
        in_dim = 256 + 128 + 128 + (256 if use_pixels else 0) + 256 + 128
        self.pre = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        # heads
        self.intent_head = nn.Linear(hidden, len(INTENTS))
        self.film = nn.Linear(len(INTENTS), 2 * hidden)
        self.action_trunk = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU())
        self.action_heads = nn.ModuleList([nn.Linear(256, n) for _, n in HEADS])
        # training-only heads
        self.value = nn.Sequential(nn.Linear(hidden + PRIV_DIM, 256), nn.ReLU(), nn.Linear(256, 1))
        self.aux = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in AUX.items()})

    def encode(self, x: dict) -> torch.Tensor:
        r = x["rays"].transpose(1, 2)
        for conv in self.ray_conv:
            r = F.relu(conv(r))
        r = F.max_pool1d(r, 4)  # (B, 64, 16)
        ray_z = F.relu(self.ray_fc(r.flatten(1)))

        e = torch.cat([x["ent_feat"].clamp(-4, 4), self.ent_type(x["ent_type"])], -1)
        e = self.ent_mlp(e)  # (B, 32, 64)
        mask = x["ent_mask"] < 0.5
        all_masked = mask.all(1, keepdim=True)
        safe_mask = mask & ~all_masked  # keep attention valid when no entity is visible
        q = self.ent_query.unsqueeze(0).expand(e.shape[0], -1, -1)
        att, _ = self.ent_attn(q, e, e, key_padding_mask=safe_mask)
        att = att * (~all_masked).float().unsqueeze(-1)
        ent_z = att.flatten(1)  # (B, 128)

        s_z = self.scalar_mlp(x["scalars"])
        parts = [ray_z, ent_z, s_z]
        if self.use_pixels:
            parts.append(self.pix(x["pix"].float() / 255.0))
        parts.append(self.map_fine(x["map_fine"].float() / 255.0))
        parts.append(self.map_coarse(x["map_coarse"].float() / 255.0))
        return self.pre(torch.cat(parts, -1))

    def core(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.gru(z, h)

    def heads(self, h: torch.Tensor):
        intent_logits = self.intent_head(h)
        gamma, beta = self.film(F.softmax(intent_logits, -1)).chunk(2, -1)
        a = self.action_trunk(h * (1 + gamma) + beta)
        return intent_logits, [head(a) for head in self.action_heads]

    def forward_step(self, x: dict, h: torch.Tensor):
        h = self.core(self.encode(x), h)
        intent_logits, logits = self.heads(h)
        return h, intent_logits, logits

    def value_fn(self, h: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        return self.value(torch.cat([h, priv], -1)).squeeze(-1)

    def initial_state(self, batch: int, device="cpu") -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)


def inference_state_dict(model: YodhaNet) -> dict:
    """Weights needed at evaluation (drops the privileged critic and auxiliary heads)."""
    return {k: v for k, v in model.state_dict().items() if not (k.startswith("value.") or k.startswith("aux."))}
