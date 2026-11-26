"""Multiprocess vectorized YodhaEnv: W worker processes x K envs each.

Observations/labels are written by workers straight into shared memory (no pickling of big arrays);
only rewards/dones/episode stats travel through the pipes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from multiprocessing import shared_memory

import numpy as np

from submission.yodha.features import INPUT_SPECS

EXTRA_SPECS = {
    "priv": ((16,), np.float32), "teacher": ((9,), np.int8), "intent": ((), np.int64),
    "aux_exit": ((), np.float32), "aux_door": ((), np.float32), "aux_dir": ((), np.int64),
}
ALL_SPECS = {**INPUT_SPECS, **EXTRA_SPECS}


def _alloc(k: int):
    shms, arrays = {}, {}
    for name, (shape, dtype) in ALL_SPECS.items():
        nbytes = int(np.prod((k,) + shape)) * np.dtype(dtype).itemsize
        shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
        shms[name] = shm
        arrays[name] = np.ndarray((k,) + shape, dtype=dtype, buffer=shm.buf)
    return shms, arrays


def _attach(names: dict, k: int):
    shms, arrays = {}, {}
    for name, (shape, dtype) in ALL_SPECS.items():
        shm = shared_memory.SharedMemory(name=names[name])
        shms[name] = shm
        arrays[name] = np.ndarray((k,) + shape, dtype=dtype, buffer=shm.buf)
    return shms, arrays


def _write(arrays: dict, i: int, inp: dict, ex: dict) -> None:
    for k in INPUT_SPECS:
        arrays[k][i] = inp[k]
    for k in EXTRA_SPECS:
        arrays[k][i] = ex[k]


def _worker(remote, env_ids, seed, tier_weights, teacher, shm_names):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from training.envs.yodha_env import YodhaEnv

    k = len(env_ids)
    shms, arrays = _attach(shm_names, k)
    envs = [YodhaEnv(i, seed, tier_weights, teacher) for i in env_ids]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                for i, e in enumerate(envs):
                    inp, ex = e.reset()
                    _write(arrays, i, inp, ex)
                remote.send(True)
            elif cmd == "step":
                rew = np.zeros(k, np.float32)
                done = np.zeros(k, np.bool_)
                episodes = []
                for i, (e, a) in enumerate(zip(envs, data)):
                    inp, r, d, ex = e.step(a)
                    _write(arrays, i, inp, ex)
                    rew[i], done[i] = r, d
                    if "episode" in ex:
                        episodes.append(ex["episode"])
                remote.send((rew, done, episodes))
            elif cmd == "tiers":
                for e in envs:
                    e.set_tier_weights(data)
                remote.send(True)
            elif cmd == "close":
                for e in envs:
                    e.close()
                for s in shms.values():
                    s.close()
                remote.close()
                return
    except Exception:
        traceback.print_exc()
        raise


class VecYodha:
    def __init__(self, n_workers: int, envs_per_worker: int, seed: int = 0,
                 tier_weights=(0.35, 0.35, 0.2, 0.1), teacher: bool = True):
        ctx = mp.get_context("spawn")
        self.k = envs_per_worker
        self.n = n_workers * envs_per_worker
        self.remotes, self.procs, self.shms, self.arrays = [], [], [], []
        for w in range(n_workers):
            shms, arrays = _alloc(self.k)
            self.shms.append(shms)
            self.arrays.append(arrays)
            parent, child = ctx.Pipe()
            ids = list(range(w * self.k, (w + 1) * self.k))
            names = {n: s.name for n, s in shms.items()}
            p = ctx.Process(target=_worker, args=(child, ids, seed, tier_weights, teacher, names), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)

    def _gather(self):
        inp = {k: np.concatenate([a[k] for a in self.arrays]) for k in INPUT_SPECS}
        ex = {k: np.concatenate([a[k] for a in self.arrays]) for k in EXTRA_SPECS}
        return inp, ex

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        for r in self.remotes:
            r.recv()
        return self._gather()

    def step(self, actions: np.ndarray):
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.k:(i + 1) * self.k]))
        res = [r.recv() for r in self.remotes]
        rew = np.concatenate([x[0] for x in res])
        done = np.concatenate([x[1] for x in res])
        episodes = [e for x in res for e in x[2]]
        inp, ex = self._gather()
        return inp, rew, done, ex, episodes

    def set_tier_weights(self, w) -> None:
        for r in self.remotes:
            r.send(("tiers", list(map(float, w))))
        for r in self.remotes:
            r.recv()

    def close(self) -> None:
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
        for shms in self.shms:
            for s in shms.values():
                s.close()
                try:
                    s.unlink()
                except FileNotFoundError:
                    pass
