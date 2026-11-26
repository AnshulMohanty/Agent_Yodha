"""Rules audit for the submission: decisions come from the network only.

Fails if submission/ imports training code, or contains planners (BFS/A*/Dijkstra/heapq/deque search)
or trigonometric aim code (atan2). EgoMap/odometry use sin/cos for geometry only."""
import ast
from pathlib import Path

SUB = Path(__file__).resolve().parents[1] / "submission"
BANNED_NAMES = {"atan2", "arctan2", "heapq", "deque", "astar", "a_star", "dijkstra", "bfs", "shortest_path"}


def _sources():
    return [p for p in SUB.rglob("*.py")]


def test_no_training_imports():
    for p in _sources():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("training", "doom_env"):
                raise AssertionError(f"{p}: imports {node.module}")
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name.split(".")[0] not in ("training",), f"{p}: imports {a.name}"


def test_no_planners_or_aimbots():
    for p in _sources():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name and name.lower() in BANNED_NAMES:
                # arctan2 on entity sin/cos is used ONLY to place seen entities into the memory map (geometry),
                # never to choose actions; the plain name atan2 / planners are banned outright.
                raise AssertionError(f"{p}:{node.lineno} uses banned name {name}")


def test_agent_contract():
    import numpy as np
    from doom_env import action_space, observation_space
    from submission.agent import DoomAgent

    ag = DoomAgent(observation_space(), action_space(), "cpu")
    ag.reset()
    for _ in range(5):
        a = ag.act(observation_space().sample())
        assert a.shape == (9,) and set(np.unique(a)) <= {0, 1}
