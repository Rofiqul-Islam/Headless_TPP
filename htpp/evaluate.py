"""Evaluation: held-out likelihood split into time / type / silence parts, type
accuracy and macro-F1, next-time error from the exact predictive distribution,
calibration (PIT), and temperature calibration on validation data (stage 4)."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from .layout import WindowDataset, token_batches
from .pipeline import Setup
from .timecode import log_survival_np, log_survival_torch
from .train import make_loader


def _n_branch_tokens(tc, depth):
    n, p = 0, 1
    for lv in range(max(min(depth, tc.L) - 1, 0)):
        p *= tc.branching[lv]
        n += p
    return n


@torch.no_grad()
def collect(model, S: Setup, split: str, branch_events: set | None = None, predict_depth: int = 0,
            use_desc: bool = False) -> dict:
    model.eval()
    tcfg = S.cfg["train"]
    opts = replace(S.opts, predict_depth=predict_depth if branch_events else 0)
    ds = WindowDataset(S.pre[split], S.wins[split], S.vocab, opts, branch_events)
    extra = None
    if branch_events and predict_depth > 0:
        nb = _n_branch_tokens(S.tc, predict_depth)
        extra = []
        for w in S.wins[split]:
            ps = S.pre[split][w.seq]
            cnt = sum((ps.sid, j) in branch_events for j in range(w.start + w.n_ctx, w.end))
            extra.append(cnt * nb)
    batches = token_batches(S.wins[split], int(tcfg.get("eval_tokens_per_batch", tcfg.get("max_tokens_per_batch", 16384))),
                            int(tcfg.get("max_batch_size", 16)), False, np.random.default_rng(0), extra)
    L = S.tc.L
    rec = {"events": [], "lvl": [[] for _ in range(L)], "mark": [], "lp_desc": [], "gaps": [],
           "gap_lvl": [[] for _ in range(L)], "branch": {}}
    for batch in make_loader(ds, batches, S.tokenizer.pad_token_id, L, model.dtype, int(tcfg.get("workers", 2))):
        o = model.compute(batch, use_desc=use_desc, keep=True)
        off = len(rec["events"])
        rec["events"].extend(batch["events"])
        for lv in range(L):
            rec["lvl"][lv].append(o["lvl_logits"][lv].numpy())
        rec["mark"].append(o["mark_logits"].numpy())
        rec["lp_desc"].append(o["lp_desc"].float().cpu().numpy())
        if batch["gaps"]:
            rec["gaps"].extend(batch["gaps"])
            for lv in range(L):
                rec["gap_lvl"][lv].append(o["gap_lvl_logits"][lv].numpy())
        for (ev, per) in o.get("branch_logits", []):
            rec["branch"][off + ev] = [p.numpy() for p in per]
    rec["lvl"] = [np.concatenate(x) for x in rec["lvl"]]
    rec["mark"] = np.concatenate(rec["mark"])
    rec["lp_desc"] = np.concatenate(rec["lp_desc"])
    rec["gap_lvl"] = [np.concatenate(x) if x else np.zeros((0, len(S.vocab.level_ids[i]))) for i, x in enumerate(rec["gap_lvl"])]
    model.train()
    return rec


def _log_softmax_np(x, T=1.0):
    """Row-wise log-softmax; rows that are all NaN (not read) stay NaN."""
    z = x / T
    out = np.full_like(z, np.nan, dtype=np.float64)
    ok = ~np.isnan(z).any(axis=1)
    if ok.any():
        zz = z[ok]
        m = zz.max(axis=1, keepdims=True)
        out[ok] = zz - m - np.log(np.exp(zz - m).sum(axis=1, keepdims=True))
    return out


def fit_temperatures(rec: dict, L: int, marks: bool = True, iters: int = 200) -> dict:
    """Stage 4: one temperature per time level (and one for the type token) fitted by
    minimising validation NLL; used for density, survival and intensity alike."""
    ev = rec["events"]
    n = len(ev)
    path = torch.tensor(np.array([e["path"] for e in ev]), dtype=torch.long)
    plen = torch.tensor([e["plen"] for e in ev], dtype=torch.long)
    k = torch.tensor([e["k"] for e in ev], dtype=torch.long)
    lvl = [torch.nan_to_num(torch.tensor(x, dtype=torch.float32)) for x in rec["lvl"]]
    mk = torch.nan_to_num(torch.tensor(rec["mark"], dtype=torch.float32))
    G = len(rec["gaps"])
    if G:
        gpath = torch.full((G, L), -1, dtype=torch.long)
        for i, g in enumerate(rec["gaps"]):
            gpath[i, : len(g["path"])] = torch.tensor(g["path"])
        gplen = torch.tensor([g["plen"] for g in rec["gaps"]], dtype=torch.long)
        glr = torch.tensor([g["log_r"] for g in rec["gaps"]], dtype=torch.float32)
        glvl = [torch.tensor(x, dtype=torch.float32) for x in rec["gap_lvl"]]
    theta = torch.zeros(L + 1, requires_grad=True)
    const = -float(np.sum([e["log_u"] for e in ev])) / max(n, 1)   # in-bin term (does not depend on T)

    def temps_of(th):
        return torch.exp(3.0 * torch.tanh(th / 3.0))      # temperatures kept in [0.05, 20]

    def objective():
        T = temps_of(theta)
        tot = torch.zeros(())
        for lv in range(L):
            lp = (lvl[lv] / T[lv]).log_softmax(-1)
            act = plen > lv
            q = lp.gather(1, path[:, lv].clamp(min=0)[:, None]).squeeze(1)
            tot = tot - (q * act).sum()
        Tk = T[L] if marks else torch.ones(())
        tot = tot - (mk / Tk).log_softmax(-1).gather(1, k[:, None]).sum()
        if G:
            gl = [(glvl[lv] / T[lv]).log_softmax(-1) for lv in range(L)]
            tot = tot - log_survival_torch(gl, gpath, gplen, glr).sum()
        return tot / max(n, 1) + const

    opt = torch.optim.LBFGS([theta], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        v = objective()
        v.backward()
        return v

    before = float(objective())
    opt.step(closure)
    after = float(objective())
    T = temps_of(theta).detach().numpy()
    if not np.isfinite(after) or after > before:
        T = np.ones(L + 1)
        after = before
    return {"time": [float(x) for x in T[:L]], "mark": float(T[L]) if marks else 1.0,
            "val_nll_before": before, "val_nll_after": after}


def macro_f1(y, p):
    labels = np.union1d(np.unique(y), np.unique(p))
    f = []
    for c in labels:
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


def compute_metrics(rec: dict, S: Setup, temps: dict | None = None, depth: int = 0, cap: float | None = None):
    tc = S.tc
    L = tc.L
    ev = rec["events"]
    n = len(ev)
    T = (temps or {}).get("time", [1.0] * L)
    Tk = (temps or {}).get("mark", 1.0)
    path = np.array([e["path"] for e in ev], dtype=np.int64)
    plen = np.array([e["plen"] for e in ev], dtype=np.int64)
    k = np.array([e["k"] for e in ev], dtype=np.int64)
    log_u = np.array([e["log_u"] for e in ev])
    log_r = np.array([e["log_r"] for e in ev])
    lps = []
    lp_time = np.zeros(n)
    for lv in range(L):
        lp = _log_softmax_np(rec["lvl"][lv], T[lv])
        act = plen > lv
        q = np.take_along_axis(np.nan_to_num(lp, nan=0.0), np.clip(path[:, lv], 0, None)[:, None], 1)[:, 0]
        lp_time += np.where(act, q, 0.0)
        lps.append(np.nan_to_num(lp, nan=0.0))
    nll_time = -(lp_time + log_u)
    lpk = _log_softmax_np(rec["mark"], Tk)
    nll_type = -np.take_along_axis(lpk, k[:, None], 1)[:, 0]
    pred_k = np.argmax(rec["mark"], axis=1)
    logS = log_survival_np(lps, path, plen, log_r)
    pit = 1.0 - np.exp(logS)
    # gaps
    G = len(rec["gaps"])
    nll_sil = np.zeros(G)
    if G:
        gpath = np.full((G, L), -1, dtype=np.int64)
        for i, g in enumerate(rec["gaps"]):
            gpath[i, : len(g["path"])] = g["path"]
        gplen = np.array([g["plen"] for g in rec["gaps"]])
        glr = np.array([g["log_r"] for g in rec["gaps"]])
        glp = [_log_softmax_np(rec["gap_lvl"][lv], T[lv]) for lv in range(L)]
        nll_sil = -log_survival_np(glp, gpath, gplen, glr)
    m = {"n_events": n, "n_gaps": G,
         "nll_time": float(nll_time.sum() / n), "nll_type": float(nll_type.sum() / n),
         "nll_sil": float(nll_sil.sum() / n)}
    m["nll"] = m["nll_time"] + m["nll_type"] + m["nll_sil"]
    m["type_acc"] = float(np.mean(pred_k == k))
    m["type_macro_f1"] = macro_f1(k, pred_k)
    srt = np.sort(np.clip(pit, 0, 1))
    m["pit_ks"] = float(np.max(np.abs(srt - (np.arange(1, n + 1) - 0.5) / n))) if n else float("nan")
    hist = np.histogram(np.clip(pit, 0, 1), bins=10, range=(0, 1))[0] / max(n, 1)
    m["pit_ece"] = float(np.mean(np.abs(hist - 0.1)))
    m["time_tokens_per_event"] = float(np.mean(plen))
    nd = np.array([e["n_desc"] for e in ev])
    if nd.sum() > 0:
        m["desc_nll_per_token"] = float(-rec["lp_desc"].sum() / nd.sum())
        m["desc_ppl"] = float(math.exp(m["desc_nll_per_token"]))
    true = [g.get("true_nll") for g in rec["gaps"]]
    if G and all(x is not None for x in true):
        m["nll_true"] = float(np.sum(true) / n)
        m["gap_to_true"] = m["nll"] - m["nll_true"]
    # predictions from the expanded token tree
    pm = np.full(n, np.nan)
    pmed = np.full(n, np.nan)
    if rec["branch"]:
        for i, per in rec["branch"].items():
            lv_lp = [_log_softmax_np(per[lv], T[lv]) for lv in range(len(per))]
            p_fine, p_tail = tc.fine_probs(lv_lp, depth=len(per))
            pm[i], pmed[i] = tc.mean_median(p_fine, p_tail, cap)
        idx = np.array(sorted(rec["branch"]))
        tau = np.array([ev[i]["tau"] for i in idx])
        m["n_pred"] = int(len(idx))
        m["rmse_mean"] = float(np.sqrt(np.mean((pm[idx] - tau) ** 2)))
        m["mae_median"] = float(np.mean(np.abs(pmed[idx] - tau)))
        m["mae_log"] = float(np.mean(np.abs(np.log1p(pmed[idx]) - np.log1p(tau))))
    df = pd.DataFrame({"seq": [e["seq"] for e in ev], "idx": [e["idx"] for e in ev], "k": k,
                       "prev_k": [e["prev_k"] for e in ev], "hist_len": [e["hist_len"] for e in ev],
                       "ctx_len": [e["ctx_len"] for e in ev], "tau": [e["tau"] for e in ev], "plen": plen,
                       "nll_time": nll_time, "nll_type": nll_type, "pred_k": pred_k, "pit": pit,
                       "pred_mean": pm, "pred_median": pmed, "lp_desc": rec["lp_desc"], "n_desc": nd})
    gdf = pd.DataFrame({"seq": [g["seq"] for g in rec["gaps"]], "gap": [g["gap"] for g in rec["gaps"]],
                        "nll_sil": nll_sil, "plen": [g["plen"] for g in rec["gaps"]]})
    return m, df, gdf


@torch.no_grad()
def text_features(model, pre) -> dict:
    """Per scored event: cosine similarity between the previous review and this event's
    review (mean input embedding of the LLM; a stratum for 'semantically related
    events'), and the length in tokens of the previous review (a proxy for text quality)."""
    E = model.embed.weight
    out = {}
    for p in pre:
        if p.desc is None:
            continue
        vec = []
        for d in p.desc:
            vec.append(E[torch.tensor(d, device=E.device)].float().mean(0) if len(d) else None)
        for j in range(max(1, p.n_hist), len(p.k)):
            a, b = vec[j - 1], vec[j]
            sim = float(torch.nn.functional.cosine_similarity(a, b, dim=0)) if a is not None and b is not None else float("nan")
            out[(p.sid, j)] = (sim, len(p.desc[j - 1]))
    return out
