# Measured facts (Phase 0, MacBook Air M5, single core, random-walk policy)

| Map | Pixels | Env init | Steps/s (1 core) |
|---|---|---|---|
| maps/two_rooms.wad (tiny grid map) | off | 0.16 s | ~3,800 |
| maps/two_rooms.wad | on | 0.18 s | ~3,800 |
| freedoom2 MAP01 (big real map) | off / on | 0.23–0.26 s | ~600 |
| freedoom2 MAP03 | off / on | 0.25 s | ~340 |

Takeaways
- Pixel buffers cost almost nothing → keep the pixel branch (needed to see GATE exit floors).
- Env init ≈ 0.2 s → load a fresh generated map every episode.
- Sensor cost grows with line count → generated maps must merge collinear edges; big Freedoom maps are ~6–10x slower.
- Validator on Mac: PASSED (2 macOS-only warnings: CPU pinning and RAM need Linux → check in Codespaces).

Environment
- Python 3.13 (uv), vizdoom 1.3.3, torch 2.15.0 (same as official image), numpy 2.5.5, MPS available.
