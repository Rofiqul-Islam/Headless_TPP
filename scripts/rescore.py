#!/usr/bin/env python
"""Score finished runs by the probability of the recorded interval of waiting times.

    python scripts/rescore.py --runs runs/rq1_bench --data data

On data with rounded or repeated timestamps, a continuous density evaluated at the
recorded value rewards whichever model makes the sharpest peaks there. This script scores
every model on the same intervals (see htpp/interval.py) and writes rescore.json next to
each run's summary.json; make_tables.py then uses it for the benchmark table.

* Headless-TPP runs (headless, digits, scratch) are reloaded from best.pt with the
  calibration temperatures stored in summary.json; no training.
* The neural TPP baselines keep no checkpoint, so they are retrained with the same seed
  (a few minutes each) and scored with the same intervals.
The step delta is chosen once per data set from its training waiting times: the recording
step when the times lie on a grid, else 1/1000 of the median waiting time. Override it
with --delta retweet=1 taobao=1e-4. Each run is also scored at 10 x delta as a check.
"""
from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from htpp.data import load_meta, load_split  # noqa: E402
from htpp.interval import cells, choose_delta, headless_interval_logp, interval_nll  # noqa: E402
from htpp.utils import get_logger, load_json, save_json, set_seed  # noqa: E402


def local_data_dir(stored: str, root: Path) -> Path:
    p = Path(stored)
    if p.exists():
        return p
    name = re.split(r"[\\/]", str(stored).rstrip("\\/"))[-1]
    for r in (root, Path("data"), ROOT / "data"):
        if (Path(r) / name / "meta.json").exists():
            return Path(r) / name
    raise FileNotFoundError(f"data folder '{name}' not found; pass --data <folder that contains it>")


def picked_events(seqs, n: int) -> set:
    """The same deterministic sample of scored test events as htpp.pipeline.pick_events,
    computed from raw sequences (so baselines are scored on exactly the same events)."""
    from htpp.utils import stable_hash
    cand = [(str(s.sid), j) for s in seqs for j in range(max(1, int(getattr(s, "n_hist", 0) or 0)), len(s.k))]
    if n is None or n <= 0 or n >= len(cand):
        return set(cand)
    cand.sort(key=lambda x: stable_hash(0, x[0], x[1]))
    return set(cand[:n])


def rescore_headless(run: Path, cfg: dict, summary: dict, deltas: list, n_events: int, max_branch: int, log) -> dict:
    from htpp.evaluate import _log_softmax_np, _n_branch_tokens, collect, compute_metrics
    from htpp.model import HeadlessTPP
    from htpp.pipeline import pick_events, setup
    set_seed(int(cfg["train"].get("seed", 1)))
    S = setup(cfg, log, splits=("train", "val", "test"))
    name_ids = S.name_ids if cfg["model"].get("mark_init", "mean") == "name" else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HeadlessTPP(cfg["model"], S.vocab, name_ids, device)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_trainable_state(ck["state"])
    temps = summary.get("temps")
    tc, L = S.tc, S.tc.L
    # expand the token tree as deep as the token budget allows (the full tree makes the scores exact)
    depth = max(d for d in range(1, L + 1) if _n_branch_tokens(tc, d) <= max_branch)
    picked = pick_events(S.pre["test"], n_events, 0)
    log.info(f"  {len(picked)} test events, token tree expanded to depth {depth} of {L}")
    rec = collect(model, S, "test", picked if depth > 1 else None, depth if depth > 1 else 0, use_desc=False)
    m, df, _ = compute_metrics(rec, S, temps, 0)
    T = (temps or {}).get("time", [1.0] * L)
    lvl = [_log_softmax_np(rec["lvl"][lv], T[lv]) for lv in range(L)]
    branch = {i: [_log_softmax_np(per[lv], T[lv]) for lv in range(len(per))] for i, per in rec["branch"].items()}
    ev = rec["events"]
    idx = np.array([i for i, e in enumerate(ev) if (e["seq"], e["idx"]) in picked], dtype=np.int64)
    path = np.full((len(ev), L), -1, dtype=np.int64)
    for i, e in enumerate(ev):
        path[i, : len(e["path"])] = e["path"]
    plen = np.array([e["plen"] for e in ev])
    tau = np.array([e["tau"] for e in ev], float)[idx]
    sub = df.iloc[idx]
    out = {"n_events": int(len(idx)), "depth": depth, "nll": m["nll"], "summary_nll": summary["metrics"]["nll"],
           "sample": {"nll_time": float(sub.nll_time.mean()), "nll_type": float(sub.nll_type.mean())},
           "nll_sil": m["nll_sil"], "interval": {}}
    for dl in deltas:
        lo, hi = cells(tau, dl)
        lp, approx = headless_interval_logp(tc, lvl, path, plen, lo, hi, branch, rows=idx)
        it = float(interval_nll(lp, lo, hi).mean())
        out["interval"][repr(dl)] = {"delta": dl, "nll_time": it, "nll": it + out["sample"]["nll_type"] + m["nll_sil"],
                                     "approx_share": float(approx.mean())}
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def rescore_baseline(run: Path, cfg: dict, summary: dict, deltas: list, n_events: int, log) -> dict:
    import pandas as pd
    from htpp.baselines import run_baseline
    set_seed(int(cfg["train"].get("seed", 1)))
    d = cfg["data"]["dir"]
    meta = load_meta(d)
    splits = {s: load_split(d, s, cfg["data"].get(f"max_{s}_seqs")) for s in ("train", "val", "test")}
    cfg.setdefault("timecode", {}).setdefault("time_resolution", meta.get("time_resolution"))
    cfg.setdefault("eval", {})["interval_deltas"] = deltas
    out_dir = run / "rescore"
    out_dir.mkdir(exist_ok=True)
    m = run_baseline(cfg, splits, int(meta["K"]), out_dir, log)
    df = pd.read_csv(out_dir / "test_events.csv.gz", dtype={"seq": str})
    picked = picked_events(splits["test"], n_events)
    sub = df[[(a, int(b)) in picked for a, b in zip(df.seq, df.idx)]]
    out = {"n_events": int(len(sub)), "nll": m["nll"], "summary_nll": summary["metrics"]["nll"],
           "sample": {"nll_time": float(sub.nll_time.mean()), "nll_type": float(sub.nll_type.mean())},
           "nll_sil": m["nll_sil"], "interval": {}, "retrained": True}
    for di, dl in enumerate(deltas):
        it = float(sub[f"nll_time_int{di}"].mean())
        out["interval"][repr(dl)] = {"delta": dl, "nll_time": it, "nll": it + out["sample"]["nll_type"] + m["nll_sil"],
                                     "approx_share": 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/rq1_bench", help="folder with the runs to rescore (searched recursively)")
    ap.add_argument("--data", default="data", help="folder that contains the data sets")
    ap.add_argument("--delta", nargs="*", default=[], help="per data set step, e.g. retweet=1 taobao=1e-4")
    ap.add_argument("--redo", action="store_true", help="rescore runs that already have rescore.json")
    ap.add_argument("--events", type=int, default=10000, help="test events scored per data set (same sample for every model; 0 = all)")
    ap.add_argument("--max_branch_tokens", type=int, default=1200, help="extra tokens per event for expanding the token tree")
    a = ap.parse_args()
    log = get_logger(None, "rescore")
    manual = {k: float(v) for k, v in (x.split("=", 1) for x in a.delta)}
    runs = sorted(p.parent for p in Path(a.runs).rglob("summary.json") if (p.parent / "config.json").exists())
    log.info(f"{len(runs)} finished runs under {a.runs}")
    steps = {}
    for run in runs:
        if (run / "rescore.json").exists() and not a.redo:
            log.info(f"skip {run} (rescore.json exists)")
            continue
        summary = load_json(run / "summary.json")
        cfg = summary.get("config") or load_json(run / "config.json")
        cfg["data"]["dir"] = str(local_data_dir(cfg["data"]["dir"], Path(a.data)))
        name = Path(cfg["data"]["dir"]).name
        if name not in steps:
            if name in manual:
                steps[name] = (manual[name], {"manual": True, "delta": manual[name]})
            else:
                tr = load_split(cfg["data"]["dir"], "train")
                taus = np.concatenate([s.tau[1:] for s in tr if len(s) > 1])
                steps[name] = choose_delta(taus)
            log.info(f"[{name}] interval step delta = {steps[name][0]:.6g}  ({steps[name][1]})")
        dl, info = steps[name]
        deltas = [dl, 10 * dl]
        log.info(f"rescoring {run}")
        if cfg.get("method", "headless") == "baseline":
            r = rescore_baseline(run, cfg, summary, deltas, a.events, log)
        else:
            if not (run / "best.pt").exists():
                log.info(f"  no best.pt in {run}; skipped")
                continue
            r = rescore_headless(run, cfg, summary, deltas, a.events, a.max_branch_tokens, log)
        r["delta_info"] = info
        diff = abs(r["nll"] - r["summary_nll"])
        if diff > 0.01:
            log.info(f"  WARNING: density NLL {r['nll']:.4f} differs from summary.json ({r['summary_nll']:.4f}) by {diff:.4f}")
        iv = r["interval"][repr(dl)]
        dens = r["sample"]["nll_time"] + r["sample"]["nll_type"] + r["nll_sil"]
        log.info(f"  on {r['n_events']} events: density NLL {dens:.4f} | interval NLL {iv['nll']:.4f} "
                 f"(delta {dl:.4g}; approximate share {iv['approx_share']:.3f})")
        save_json(r, run / "rescore.json")


if __name__ == "__main__":
    main()
