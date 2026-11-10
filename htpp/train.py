"""Training: the four stages of Section 3 (event grammar, next event, silence, and
calibration, which is done after training in evaluate.py)."""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .layout import WindowDataset, collate, token_batches
from .pipeline import Setup, grammar_windows
from .utils import Timer


def make_loader(ds, batches, pad_id, n_levels, dtype, workers=2):
    return torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=workers,
                                       collate_fn=partial(collate, pad_id=pad_id, n_levels=n_levels, dtype=dtype),
                                       persistent_workers=False)


@torch.no_grad()
def evaluate_nll(model, S: Setup, split: str, losses: set, lambda_d=0.0, max_batches=None) -> dict:
    """Held-out NLL per scored event (time includes the in-bin term log u_B)."""
    model.eval()
    tc = S.cfg["train"]
    ds = WindowDataset(S.pre[split], S.wins[split], S.vocab, S.opts)
    rng = np.random.default_rng(0)
    batches = token_batches(S.wins[split], int(tc.get("eval_tokens_per_batch", tc.get("max_tokens_per_batch", 16384))),
                            int(tc.get("max_batch_size", 16)), False, rng)
    if max_batches:
        batches = batches[: int(max_batches)]
    tot = {"time": 0.0, "type": 0.0, "sil": 0.0, "desc": 0.0, "n": 0, "n_desc": 0}
    for batch in make_loader(ds, batches, S.tokenizer.pad_token_id, S.tc.L, model.dtype, int(tc.get("workers", 2))):
        o = model.compute(batch, use_desc=("desc" in losses))
        tot["time"] -= float(o["lp_time"].sum()) + float(batch["ev_log_u"].sum())
        tot["type"] -= float(o["lp_type"].sum())
        tot["sil"] -= float(o["logS_gap"].sum())
        tot["desc"] -= float(o["lp_desc"].sum())
        tot["n"] += batch["n_events"]
        tot["n_desc"] += sum(e["n_desc"] for e in batch["events"])
    n = max(tot["n"], 1)
    res = {"nll_time": tot["time"] / n, "nll_type": tot["type"] / n, "nll_sil": tot["sil"] / n, "n_events": tot["n"]}
    res["nll"] = res["nll_time"] + res["nll_type"] + res["nll_sil"]
    res["select"] = (res["nll_time"] if "time" in losses else 0) + (res["nll_type"] if "type" in losses else 0) + \
                    (res["nll_sil"] if "sil" in losses else 0)
    if tot["n_desc"]:
        res["desc_ppl"] = math.exp(tot["desc"] / tot["n_desc"])
    model.train()
    return res


def train(model, S: Setup, log, out_dir: str | Path) -> dict:
    tcfg = S.cfg["train"]
    out_dir = Path(out_dir)
    seed = int(tcfg.get("seed", 1))
    rng = np.random.default_rng(seed)
    lambda_d = float(tcfg.get("lambda_d", 0.0))
    stages = tcfg["stages"]
    final_losses = set(stages[-1]["losses"])
    if lambda_d <= 0:
        final_losses.discard("desc")
    pad = S.tokenizer.pad_token_id
    L = S.tc.L
    max_tok = int(tcfg.get("max_tokens_per_batch", 16384))
    max_bs = int(tcfg.get("max_batch_size", 16))
    accum = int(tcfg.get("grad_accum", 1))
    workers = int(tcfg.get("workers", 2))

    groups = [{"params": model.lora_parameters(), "lr": float(tcfg.get("lr", 2e-4)),
               "weight_decay": float(tcfg.get("weight_decay", 0.0))},
              {"params": model.new_token_parameters(), "lr": float(tcfg.get("lr_new", 1e-3)), "weight_decay": 0.0}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)

    train_ds = WindowDataset(S.pre["train"], S.wins["train"], S.vocab, S.opts)
    n_batches = len(token_batches(S.wins["train"], max_tok, max_bs, False, rng))
    plan = []
    for st in stages:
        steps = int(st["steps"]) if "steps" in st else int(math.ceil(float(st.get("epochs", 1)) * n_batches / accum))
        plan.append(steps)
    total = sum(plan)
    warm = max(1, int(float(tcfg.get("warmup_frac", 0.03)) * total))
    min_ratio = float(tcfg.get("min_lr_ratio", 0.1))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    log.info(f"training: {total} optimizer steps in stages {[(s['name'], n) for s, n in zip(stages, plan)]}, "
             f"{n_batches} train batches per epoch, trainable params "
             f"{sum(p.numel() for g in groups for p in g['params']) / 1e6:.2f}M")

    syn = None
    if any(float(st.get("synthetic_frac", 0)) > 0 for st in stages):
        n_syn = int(tcfg.get("synthetic_sequences", 2000))
        pre_s, wins_s, opt_s = grammar_windows(S, n_syn, seed)
        syn = (WindowDataset(pre_s, wins_s, S.vocab, opt_s), wins_s)

    def batch_stream(ds, wins):
        while True:
            for b in make_loader(ds, token_batches(wins, max_tok, max_bs, True, rng), pad, L, model.dtype, workers):
                yield b

    main_it = batch_stream(train_ds, S.wins["train"])
    syn_it = batch_stream(*syn) if syn else None
    best = {"select": float("inf")}
    step = 0
    timer = Timer()
    eval_every = int(tcfg.get("eval_every", 500))
    patience = int(tcfg.get("patience", 6))
    history = []
    model.train()
    for st, n_steps in zip(stages, plan):
        losses = set(st["losses"])
        if lambda_d <= 0:
            losses.discard("desc")
        model.set_lora_trainable(st.get("train", "all") != "new_tokens")
        frac = float(st.get("synthetic_frac", 0.0))
        bad = 0
        log.info(f"stage {st['name']}: {n_steps} steps, losses {sorted(losses)}, train {st.get('train', 'all')}")
        run = {"loss": 0.0, "n": 0}
        for i in range(n_steps):
            for _ in range(accum):
                use_syn = syn_it is not None and rng.random() < frac
                batch = next(syn_it if use_syn else main_it)
                loss, parts = model.loss(batch, losses if not use_syn else (losses & {"time", "type"}), lambda_d)
                (loss / accum).backward()
                run["loss"] += float(loss)
                run["n"] += 1
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], float(tcfg.get("clip", 1.0)))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % int(tcfg.get("log_every", 50)) == 0:
                log.info(f"step {step}/{total} [{st['name']}] loss {run['loss'] / max(run['n'], 1):.4f} "
                         f"lr {sched.get_last_lr()[0]:.2e} {timer():.0f}s")
                run = {"loss": 0.0, "n": 0}
            last = i == n_steps - 1
            if st["name"] != "grammar" and (step % eval_every == 0 or last):
                v = evaluate_nll(model, S, "val", final_losses, lambda_d, tcfg.get("eval_max_batches"))
                v["step"] = step
                v["stage"] = st["name"]
                history.append(v)
                improved = v["select"] < best["select"] - 1e-4
                log.info(f"  val nll {v['nll']:.4f} (time {v['nll_time']:.4f} type {v['nll_type']:.4f} "
                         f"sil {v['nll_sil']:.4f}) select {v['select']:.4f}{' *' if improved else ''}")
                if improved:
                    best = dict(v)
                    torch.save({"state": model.trainable_state(), "val": v}, out_dir / "best.pt")
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience and "epochs" in st:
                        log.info(f"  early stop in stage {st['name']}")
                        break
    if not (out_dir / "best.pt").exists():
        torch.save({"state": model.trainable_state(), "val": None}, out_dir / "best.pt")
    ck = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_trainable_state(ck["state"])
    log.info(f"loaded best checkpoint (val {best.get('nll', float('nan')):.4f}) after {timer():.0f}s")
    return {"best_val": best, "history": history, "steps": step, "train_seconds": timer()}
