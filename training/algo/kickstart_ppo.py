"""Kickstarted recurrent PPO for YODHA (our own implementation; PPO details follow CleanRL / "37 details").

Loss = ppo_coef * (PPO clip + entropy) + vf * value + λ_ks * CE(student, teacher)   (kickstarting)
       + intent CE + auxiliary losses.
Schedule by environment steps:
  phase A  (DAgger warm-up): λ_ks = 1, PPO off, teacher executes a fraction β (0.5 → 0) of 16-step chunks
  phase B  (kickstart):      λ_ks 1 → 0.1, PPO on, student acts
  phase C  (fine-tune):      λ_ks = final_ks, PPO on
The student always gets the teacher's label for every visited state, so this is DAgger + RL.

Usage: uv run python -m training.algo.kickstart_ppo --config training/configs/mac.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from submission.yodha.features import INPUT_SPECS
from submission.yodha.model import HEADS, YodhaNet, buttons_to_factored, factored_to_buttons, inference_state_dict
from training.envs.vec import VecYodha

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = dict(
    run_name="dev", n_workers=8, envs_per_worker=4, rollout=128, seq_len=32, epochs=4, minibatches=4,
    lr=3e-4, gamma=0.995, gae_lambda=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, ent_coef_final=0.003,
    max_grad=0.5, total_steps=60_000_000, device="auto", hidden=512, use_pixels=True,
    phase_a_end=6_000_000, phase_b_end=25_000_000, beta_start=0.5, final_ks=0.05, intent_coef=0.5,
    aux_coef=0.2, ckpt_every=10, seed=0, init_ckpt=None, tier_weights=[0.3, 0.3, 0.25, 0.15], max_hours=100.0,
    reset_optimizer=False,
)


class RunningStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = float(x.mean()), float(x.var()), x.size
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return math.sqrt(self.var) + 1e-8


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_t(arrs: dict, device) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arrs.items()}


def schedule(cfg, steps: int) -> dict:
    a, b = cfg["phase_a_end"], cfg["phase_b_end"]
    if steps < a:
        frac = steps / a
        return dict(phase="A", ks=1.0, ppo=0.0, beta=cfg["beta_start"] * (1 - frac))
    if steps < b:
        frac = (steps - a) / (b - a)
        return dict(phase="B", ks=1.0 - 0.9 * frac, ppo=1.0, beta=0.0)
    return dict(phase="C", ks=cfg["final_ks"], ppo=1.0, beta=0.0)


def head_logp_entropy(logits: list[torch.Tensor], idx: torch.Tensor):
    logp, ent = 0.0, 0.0
    for i, lg in enumerate(logits):
        lp = F.log_softmax(lg.float(), -1)
        logp = logp + lp.gather(-1, idx[..., i:i + 1]).squeeze(-1)
        ent = ent + -(lp.exp() * lp).sum(-1)
    return logp, ent


def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = pick_device(cfg["device"])
    run_dir = ROOT / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tb"))
    except Exception:  # pragma: no cover
        writer = None

    net = YodhaNet(hidden=cfg["hidden"], use_pixels=cfg["use_pixels"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5)
    steps, it0 = 0, 0
    if cfg.get("init_ckpt"):
        ck = torch.load(cfg["init_ckpt"], map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"], strict=False)
        if "opt" in ck and not cfg.get("reset_optimizer"):
            try:
                opt.load_state_dict(ck["opt"])
            except Exception:
                pass
        steps = int(ck.get("steps", 0)) if not cfg.get("reset_steps") else 0
        print(f"loaded {cfg['init_ckpt']} at {steps} steps")

    E = cfg["n_workers"] * cfg["envs_per_worker"]
    T, L = cfg["rollout"], cfg["seq_len"]
    vec = VecYodha(cfg["n_workers"], cfg["envs_per_worker"], cfg["seed"], cfg["tier_weights"])
    inp, ex = vec.reset()
    h = net.initial_state(E, device)
    starts_next = np.ones(E, np.float32)
    rstd, ret_acc = RunningStd(), np.zeros(E)
    beta_mask = np.zeros(E, np.bool_)
    ep_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
    tier_success = np.full(4, 0.3)
    t_start = time.time()

    buf = {k: np.zeros((T, E) + shape, dtype) for k, (shape, dtype) in INPUT_SPECS.items()}
    n_iters = int(cfg["total_steps"] // (E * T))
    for it in range(it0, n_iters):
        sch = schedule(cfg, steps)
        ent_coef = cfg["ent_coef"] + (cfg["ent_coef_final"] - cfg["ent_coef"]) * min(1.0, steps / cfg["total_steps"])
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * max(0.1, 1 - steps / cfg["total_steps"])
        H0_dev = torch.zeros((T, E, cfg["hidden"]), device=device)
        ACT = np.zeros((T, E, len(HEADS)), np.int64)
        TEA = np.zeros((T, E, len(HEADS)), np.int64)
        LOGP = np.zeros((T, E), np.float32)
        VAL = np.zeros((T, E), np.float32)
        REW = np.zeros((T, E), np.float32)
        DONE = np.zeros((T, E), np.float32)
        START = np.zeros((T, E), np.float32)
        BYT = np.zeros((T, E), np.float32)  # executed by teacher
        INT = np.zeros((T, E), np.int64)
        PRIV = np.zeros((T, E, ex["priv"].shape[-1]), np.float32)
        AUXE = np.zeros((T, E), np.float32)
        AUXD = np.zeros((T, E), np.float32)
        AUXF = np.zeros((T, E), np.int64)
        episodes = []
        t_roll = time.time()
        tm_inf = tm_env = 0.0
        net.eval()
        for t in range(T):
            if t % 16 == 0:
                beta_mask = np.random.rand(E) < sch["beta"]
            _t0 = time.perf_counter()
            for k in buf:
                buf[k][t] = inp[k]
            START[t] = starts_next
            with torch.no_grad():
                hin = h * torch.from_numpy(1 - starts_next).to(device).unsqueeze(-1)
                H0_dev[t] = hin
                x = to_t(inp, device)
                hn, _, logits = net.forward_step(x, hin)
                val = net.value_fn(hn, torch.from_numpy(ex["priv"]).to(device))
                # Gumbel-max sampling on device (one sync per step instead of many)
                student = torch.stack([(lg.float() - torch.log(-torch.log(torch.rand_like(lg.float()).clamp_min(1e-9))))
                                       .argmax(-1) for lg in logits], -1)
                teacher = buttons_to_factored(torch.from_numpy(ex["teacher"]).to(device))
                execd = torch.where(torch.from_numpy(beta_mask).to(device).unsqueeze(-1), teacher, student)
                logp, _ = head_logp_entropy(logits, execd)
                packed = torch.cat([execd.float(), teacher.float(), logp[:, None], val[:, None]], -1).cpu().numpy()
            nh = len(HEADS)
            ACT[t], TEA[t] = packed[:, :nh].astype(np.int64), packed[:, nh:2 * nh].astype(np.int64)
            LOGP[t], VAL[t] = packed[:, 2 * nh], packed[:, 2 * nh + 1]
            execd_np = ACT[t]
            BYT[t] = beta_mask
            INT[t], PRIV[t] = ex["intent"], ex["priv"]
            AUXE[t], AUXD[t], AUXF[t] = ex["aux_exit"], ex["aux_door"], ex["aux_dir"]
            buttons = factored_to_buttons(torch.from_numpy(execd_np)).numpy()
            _t1 = time.perf_counter()
            inp, rew, done, ex, eps = vec.step(buttons)
            tm_inf += _t1 - _t0
            tm_env += time.perf_counter() - _t1
            ret_acc = ret_acc * cfg["gamma"] + rew
            rstd.update(ret_acc)
            ret_acc[done] = 0.0
            REW[t], DONE[t] = rew, done
            starts_next = done.astype(np.float32)
            h = hn
            episodes += eps
        H0 = H0_dev.cpu().numpy()
        roll_s = time.time() - t_roll
        steps += E * T

        # bootstrap
        with torch.no_grad():
            hin = h * torch.from_numpy(1 - starts_next).to(device).unsqueeze(-1)
            hn, _, _ = net.forward_step(to_t(inp, device), hin)
            next_val = net.value_fn(hn, torch.from_numpy(ex["priv"]).to(device)).cpu().numpy()
        rew_n = np.clip(REW / rstd.std, -10, 10)
        adv = np.zeros_like(REW)
        last = np.zeros(E, np.float32)
        for t in reversed(range(T)):
            nv = next_val if t == T - 1 else VAL[t + 1]
            nonterm = 1.0 - DONE[t]
            delta = rew_n[t] + cfg["gamma"] * nv * nonterm - VAL[t]
            last = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterm * last
            adv[t] = last
        RET = adv + VAL

        # ---- update over sequence chunks ----
        net.train()
        t_upd = time.time()
        chunk_starts = [(e, t0) for e in range(E) for t0 in range(0, T, L)]
        n_mb = cfg["minibatches"]
        stats = defaultdict(list)
        for _ in range(cfg["epochs"]):
            perm = np.random.permutation(len(chunk_starts))
            for mb in np.array_split(perm, n_mb):
                es = np.array([chunk_starts[i][0] for i in mb])
                ts = np.array([chunk_starts[i][1] for i in mb])
                tt = ts[:, None] + np.arange(L)[None]  # (B, L)
                ee = es[:, None].repeat(L, 1)
                gather = lambda a: torch.from_numpy(np.ascontiguousarray(a[tt, ee].swapaxes(0, 1))).to(device)  # noqa: E731
                xs = {k: gather(buf[k]) for k in buf}
                hcur = torch.from_numpy(H0[ts, es]).to(device)
                st_ = gather(START)
                privs = gather(PRIV)
                B = len(mb)
                # encoders don't depend on the recurrent state: run them once over all L*B steps
                z = net.encode({k: v.reshape((L * B,) + v.shape[2:]) for k, v in xs.items()}).reshape(L, B, -1)
                hs = []
                for l in range(L):
                    if l > 0:
                        hcur = hcur * (1 - st_[l]).unsqueeze(-1)
                    hcur = net.core(z[l], hcur)
                    hs.append(hcur)
                hall = torch.stack(hs).reshape(L * B, -1)
                il_all, lg_all = net.heads(hall)
                outs_int = [il_all.reshape(L, B, -1)]
                value = net.value_fn(hall, privs.reshape(L * B, -1)).reshape(L, B)
                aux_all = {k: m(hall).reshape(L, B, -1) for k, m in net.aux.items()}
                logits = [lg.reshape(L, B, -1) for lg in lg_all]
                act, tea = gather(ACT), gather(TEA)
                new_logp, ent = head_logp_entropy(logits, act)
                old_logp, advs, rets = gather(LOGP), gather(adv), gather(RET)
                byt = gather(BYT)
                student_mask = 1 - byt
                advn = (advs - advs.mean()) / (advs.std() + 1e-8)
                ratio = torch.exp(new_logp - old_logp)
                pg = torch.max(-advn * ratio, -advn * ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]))
                pg_loss = (pg * student_mask).sum() / student_mask.sum().clamp(min=1)
                v_loss = 0.5 * ((value - rets) ** 2).mean()
                ks = sum(F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tea[..., i].reshape(-1))
                         for i, lg in enumerate(logits))
                il = outs_int[0]
                int_loss = F.cross_entropy(il.reshape(-1, il.shape[-1]).float(), gather(INT).reshape(-1))
                aux_e = aux_all["exit_visible"].squeeze(-1)
                aux_d = aux_all["door_ahead"].squeeze(-1)
                aux_f = aux_all["frontier_dir"]
                aux_loss = (F.binary_cross_entropy_with_logits(aux_e.float(), gather(AUXE))
                            + F.binary_cross_entropy_with_logits(aux_d.float(), gather(AUXD))
                            + F.cross_entropy(aux_f.reshape(-1, 8).float(), gather(AUXF).reshape(-1), ignore_index=-1))
                if torch.isnan(aux_loss):
                    aux_loss = torch.zeros((), device=device)
                ent_m = ent.mean()
                loss = (sch["ppo"] * (pg_loss - ent_coef * ent_m) + cfg["vf_coef"] * v_loss + sch["ks"] * ks
                        + cfg["intent_coef"] * int_loss + cfg["aux_coef"] * aux_loss)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad"])
                opt.step()
                with torch.no_grad():
                    agree = torch.stack([(lg.argmax(-1) == tea[..., i]).float().mean() for i, lg in enumerate(logits)]).mean()
                    stats["pg"].append(pg_loss.item())
                    stats["v"].append(v_loss.item())
                    stats["ks"].append(ks.item())
                    stats["intent"].append(int_loss.item())
                    stats["aux"].append(aux_loss.item())
                    stats["ent"].append(ent_m.item())
                    stats["agree"].append(agree.item())
                    stats["kl"].append((old_logp - new_logp).mean().item())
        upd_s = time.time() - t_upd

        # ---- logging / curriculum ----
        for epi in episodes:
            ep_hist[int(epi["tier"])].append(epi)
        line = []
        for tier in (1, 2, 3, 4):
            hist = ep_hist[tier]
            if not hist:
                continue
            m = {k: float(np.mean([e[k] for e in hist])) for k in ("exit", "dead", "coverage", "return")}
            keys = sum(e["keys"] for e in hist) / max(1.0, sum(e["n_keys"] for e in hist))
            kills = sum(e["kills"] for e in hist) / max(1.0, sum(e["monsters"] for e in hist))
            tier_success[tier - 1] = m["exit"]
            line.append(f"T{tier} exit {m['exit']:.2f} dead {m['dead']:.2f} keys {keys:.2f} kills {kills:.2f} cov {m['coverage']:.2f}")
            if writer:
                for k, v in dict(m, keys=keys, kills=kills).items():
                    writer.add_scalar(f"tier{tier}/{k}", v, steps)
        lp = 1.0 + 3.0 * tier_success * (1 - tier_success)  # learning progress: favour 20-80% success
        w = np.array(cfg["tier_weights"]) * lp
        vec.set_tier_weights(w / w.sum())
        sps = E * T / (roll_s + upd_s)
        ms = {k: float(np.mean(v)) for k, v in stats.items()}
        print(f"[{it}] steps {steps/1e6:.2f}M phase {sch['phase']} ks {sch['ks']:.2f} beta {sch['beta']:.2f} "
              f"sps {sps:.0f} (roll {roll_s:.1f}s [inf {tm_inf:.1f} env {tm_env:.1f}] upd {upd_s:.1f}s) agree {ms['agree']:.3f} ks {ms['ks']:.2f} "
              f"v {ms['v']:.3f} ent {ms['ent']:.2f} | " + " | ".join(line), flush=True)
        if writer:
            for k, v in ms.items():
                writer.add_scalar(f"train/{k}", v, steps)
            writer.add_scalar("train/sps", sps, steps)
            writer.add_scalar("train/ks_coef", sch["ks"], steps)
        if (it + 1) % cfg["ckpt_every"] == 0 or it == n_iters - 1:
            save(net, opt, cfg, steps, run_dir)
        if (time.time() - t_start) / 3600 > cfg["max_hours"]:
            print("max_hours reached")
            save(net, opt, cfg, steps, run_dir)
            break
    vec.close()


def save(net, opt, cfg, steps, run_dir: Path) -> None:
    ck = {"model": net.state_dict(), "opt": opt.state_dict(), "steps": steps, "cfg": cfg}
    torch.save(ck, run_dir / "latest.pt")
    numbered = run_dir / f"ckpt_{steps // 1_000_000:04d}M.pt"
    if not numbered.exists():  # keep one numbered checkpoint per million steps
        torch.save(ck, numbered)
    exp = run_dir / "export"
    exp.mkdir(exist_ok=True)
    torch.save({k: v.cpu() for k, v in inference_state_dict(net).items()}, exp / "yodha.pt")
    (exp / "config.json").write_text(json.dumps({"hidden": cfg["hidden"], "use_pixels": cfg["use_pixels"],
                                                 "temperature": 0.0, "steps": steps}))
    print(f"saved checkpoint at {steps} steps → {run_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value")
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(yaml.safe_load(Path(a.config).read_text()) or {})
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    train(cfg)


if __name__ == "__main__":
    main()
