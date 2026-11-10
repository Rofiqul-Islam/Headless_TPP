#!/usr/bin/env python
"""Collect all finished runs and write the paper's result tables (LaTeX) and figures.

  python scripts/make_tables.py --runs runs --out paper_tables

Upload the files in paper_tables/ to the Overleaf folder `generated/`; the paper
includes each table automatically once its file exists. Numbers are means over seeds
with standard errors; NLL is in nats per scored event (lower is better) and a gain
is a reduction in NLL (higher is better).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from htpp.utils import load_json, read_jsonl  # noqa: E402

COND_NAMES = {"no_text": "NO-TEXT", "names": "NAMES", "scrambled": "SCRAMBLED", "full": "FULL",
              "shuffled": "SHUFFLED", "text_only": "TEXT-ONLY"}
COMPARISONS = [("full", "no_text", "FULL vs.\\ NO-TEXT", "total value of text"),
               ("names", "scrambled", "NAMES vs.\\ SCRAMBLED", "meaning of the type name"),
               ("full", "shuffled", "FULL vs.\\ SHUFFLED", "event-specific detail"),
               ("text_only", "full", "TEXT-ONLY vs.\\ FULL", "text replacing the label"),
               ("scrambled", "no_text", "SCRAMBLED vs.\\ NO-TEXT", "format only (expect 0)")]


DATA_ROOT: Path | None = None   # set with --data; used when a run was made on another computer


def data_dir(cfg: dict) -> Path:
    """The run's data folder. Runs copied from another computer store that computer's path,
    so fall back to <--data>/<folder name> (default: ./data) when the stored path is missing."""
    stored = str(cfg["data"]["dir"])
    p = Path(stored)
    if p.exists():
        return p
    name = re.split(r"[\\/]", stored.rstrip("\\/"))[-1]
    for root in ([DATA_ROOT] if DATA_ROOT else []) + [Path("data"), ROOT / "data"]:
        q = Path(root) / name
        if (q / "meta.json").exists():
            return q
    raise FileNotFoundError(
        f"data folder not found: {stored}\n  (this run was probably made on another computer). Copy the folder "
        f"'{name}' (meta.json and train.jsonl are enough) into ./data/ or pass --data <folder that contains '{name}'>.")


def load_runs(root: Path) -> list[dict]:
    runs = []
    for p in sorted(root.rglob("summary.json")):
        s = load_json(p)
        s["dir"] = str(p.parent)
        runs.append(s)
    return runs


def ms(xs):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if len(xs) == 0:
        return np.nan, np.nan, 0
    se = xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else np.nan
    return float(xs.mean()), float(se), len(xs)


def fmt(m, se=np.nan, d=3, sign=False):
    if m is None or not np.isfinite(m):
        return "--"
    if sign and abs(m) < 0.5 * 10 ** (-d):
        m, sign = 0.0, False
    s = f"{m:+.{d}f}" if sign else f"{m:.{d}f}"
    s = s.replace("-", "$-$", 1) if s.startswith("-") else s
    if np.isfinite(se):
        s += f"{{\\tiny$\\pm${se:.{d}f}}}"
    return s


def fmt_auto(m, d=3):
    """Fewer decimals for large numbers (e.g. Retweet waiting times in seconds)."""
    if m is None or not np.isfinite(m):
        return "--"
    a = abs(m)
    s = f"{m:.0f}" if a >= 1000 else f"{m:.1f}" if a >= 100 else f"{m:.2f}" if a >= 10 else f"{m:.{d}f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def seeds_note(counts, se=True):
    counts = sorted({c for c in counts if c})
    if not counts:
        return ""
    n = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}--{counts[-1]}"
    return f"Means over {n} seeds" + ("" if counts[-1] < 2 or not se else "; $\\pm$: standard error") + "."


DS_NAMES = {"retweet": "Retweet", "stackoverflow": "StackOverflow", "taobao": "Taobao", "amazon": "Amazon",
            "hawkes_short": "Hawkes, short windows", "taobao_cut": "Taobao, cut at random times",
            "amazon_cut": "Amazon, cut at random times"}
DS_ORDER = ["retweet", "stackoverflow", "taobao", "amazon", "hawkes_short", "taobao_cut", "amazon_cut"]


def order_ds(names):
    return sorted(names, key=lambda x: (DS_ORDER.index(x) if x in DS_ORDER else len(DS_ORDER), x))


def pick(runs, **tags):
    return [r for r in runs if all(r["tags"].get(k) == v for k, v in tags.items())]


def metric(rs, key, raw=False):
    return ms([(r["metrics_raw"] if raw else r["metrics"]).get(key) for r in rs])


def write(out: Path, name: str, tex: str, results: dict, key: str, data):
    (out / f"{name}.tex").write_text(tex, encoding="utf-8")
    results[key] = data
    print(f"wrote {out / name}.tex")


# ---------------------------------------------------------------------------- RQ1
def table_hawkes(runs, out, R):
    rows = [("hier8x8x8", "Headless-TPP, $8{\\times}8{\\times}8$ tokens"), ("hier32x16", "Headless-TPP, $32{\\times}16$"),
            ("flat512", "Headless-TPP, one level (512)"), ("digits", "Digit strings (LLMTime-style)"),
            ("lnm_gru", "GRU + log-normal mixture"), ("lnm_transformer", "Transformer + log-normal mix.")]
    lines, data, counts = [], {}, []
    for var, label in rows:
        rs = pick(runs, experiment="rq1_hawkes", variant=var)
        if not rs:
            continue
        nll, gap, tok, ks = (metric(rs, k) for k in ("nll", "gap_to_true", "time_tokens_per_event", "pit_ks"))
        counts.append(nll[2])
        data[var] = {"nll": nll, "gap_to_true": gap, "tokens": tok, "pit_ks": ks}
        lines.append(f"{label} & {fmt(nll[0], nll[1])} & {fmt(gap[0], gap[1])} & {fmt(tok[0], d=0) if np.isfinite(tok[0]) else '--'} & {fmt(ks[0], ks[1])}\\\\")
    if not lines:
        return
    true = metric(pick(runs, experiment="rq1_hawkes"), "nll_true")[0]
    tex = ("\\begin{table}[t]\n\\caption{E1a: synthetic Hawkes data with a known likelihood. Held-out NLL (nats per event; "
           f"the true model scores {fmt(true)}), gap to the true NLL, tokens per waiting time, and PIT error (KS distance). "
           f"{seeds_note(counts)}}}\n\\label{{tab:rq1hawkes}}\n"
           "\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
           "Model & NLL & Gap to true & Tokens & PIT KS\\\\\n\\midrule\n" + "\n".join(lines) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(out, "rq1_hawkes", tex, R, "rq1_hawkes", data)


def table_bench(runs, out, R):
    dsets = order_ds({r["tags"]["dataset"] for r in pick(runs, experiment="rq1_bench")})
    if not dsets:
        return
    rows = [("headless", False, "Headless-TPP (ours)"), ("headless", True, "\\quad without calibration (stage 4)"),
            ("digits", False, "\\quad digit strings instead of time tokens"),
            ("scratch", False, "\\quad no language pre-training (0.5B)"),
            ("lnm_gru", False, "GRU + log-normal mixture"), ("lnm_transformer", False, "Transformer + log-normal mixture")]
    vals, data, counts = {}, defaultdict(dict), []
    for var, raw, label in rows:
        for ds in dsets:
            rs = pick(runs, experiment="rq1_bench", dataset=ds, variant=var)
            v = {k: metric(rs, m, raw) for k, m in (("nll", "nll"), ("rmse", "rmse_mean"), ("acc", "type_acc"))}
            counts.append(v["nll"][2])
            vals[(var, raw, ds)] = v
            data[f"{var}{'_raw' if raw else ''}"][ds] = v
    # best value per column (bold), ignoring the "without calibration" row
    best = {}
    for ds in dsets:
        for k, lower in (("nll", True), ("rmse", True), ("acc", False)):
            xs = [vals[(var, raw, ds)][k][0] for var, raw, _ in rows if not raw]
            xs = [x for x in xs if np.isfinite(x)]
            if xs:
                best[(ds, k)] = min(xs) if lower else max(xs)
    lines = []
    for var, raw, label in rows:
        cells = []
        for ds in dsets:
            for k in ("nll", "rmse", "acc"):
                m = vals[(var, raw, ds)][k][0]
                c = fmt_auto(m)
                if not raw and np.isfinite(m) and (ds, k) in best and fmt_auto(m) == fmt_auto(best[(ds, k)]):
                    c = f"\\textbf{{{c}}}"
                cells.append(c)
        lines.append(label + " & " + " & ".join(cells) + "\\\\")
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS_NAMES.get(ds, ds)}}}" for ds in dsets)
    sub = " & ".join(["NLL & RMSE & Acc."] * len(dsets))
    cm = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(dsets)))
    tex = ("\\begin{table*}[t]\n\\caption{E1b: benchmarks. Held-out NLL (nats per event, lower is better), RMSE of the "
           "predicted mean waiting time (capped at the longest training wait for every model; in each data set's time unit), and type "
           "accuracy. All models score the same events in the same time unit. Best value per column in bold. "
           f"{seeds_note(counts, se=False)}}}\n\\label{{tab:rq1bench}}\n\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}\n"
           f"\\begin{{tabular}}{{@{{}}l{'ccc' * len(dsets)}@{{}}}}\n\\toprule\n & {head}\\\\\n{cm}\n & {sub}\\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write(out, "rq1_bench", tex, R, "rq1_bench", data)


def table_silence(runs, out, R):
    dsets = order_ds({r["tags"]["dataset"] for r in pick(runs, experiment="rq1_silence")})
    lines, data, counts = [], {}, []
    for ds in dsets:
        block = []
        for var, label in (("with_sil", "with"), ("no_sil", "without")):
            rs = pick(runs, experiment="rq1_silence", dataset=ds, variant=var)
            if not rs:
                continue
            nll, sil, gap, ks, mae = (metric(rs, k) for k in ("nll", "nll_sil", "gap_to_true", "pit_ks", "mae_median"))
            counts.append(nll[2])
            data[f"{ds}/{var}"] = {"nll": nll, "nll_sil": sil, "gap_to_true": gap, "pit_ks": ks, "mae_median": mae}
            block.append(f"\\quad {label} & {fmt(nll[0], nll[1])} & {fmt(sil[0], sil[1])} & "
                         f"{fmt(gap[0], gap[1])} & {fmt(ks[0], ks[1])} & {fmt(mae[0], mae[1])}\\\\")
        if block:
            lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{DS_NAMES.get(ds, ds)}}}}}\\\\")
            lines += block
    if not lines:
        return
    tex = ("\\begin{table}[t]\n\\caption{E1c: the same model trained with and without the survival term for the final "
           "quiet gap. NLL and its silence part (nats per event), gap to the true NLL (synthetic data only), PIT error "
           f"of the observed events, and absolute error of the median waiting time. {seeds_note(counts)}}}\n\\label{{tab:rq1silence}}\n"
           "\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           "Survival term & NLL & Silence & Gap & PIT KS & MAE\\\\\n\\midrule\n" + "\n".join(lines) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(out, "rq1_silence", tex, R, "rq1_silence", data)


# ---------------------------------------------------------------------------- RQ3
def _by_seed(rs):
    return {r["tags"]["seed"]: r for r in rs}


def _gaps(run_dir, unseen=False):
    p = Path(run_dir) / ("test_unseen_gaps.csv.gz" if unseen else "test_gaps.csv.gz")
    return pd.read_csv(p) if p.exists() else None


def paired_users(A: dict, B: dict, unseen=False):
    """Per-user gains of run A over run B (B's NLL minus A's), averaged over the seeds both
    have: DataFrame with columns seq, n (events), time, type, sil."""
    per_seed = []
    for s in sorted(set(A) & set(B)):
        a, b = _events(A[s]["dir"], unseen), _events(B[s]["dir"], unseen)
        if a is None or b is None:
            continue
        m = a.merge(b, on=["seq", "idx"], suffixes=("_a", "_b"))
        u = m.groupby("seq").agg(n=("idx", "size"), time=("nll_time_b", "sum"), type=("nll_type_b", "sum"))
        ua = m.groupby("seq").agg(time=("nll_time_a", "sum"), type=("nll_type_a", "sum"))
        u["time"] -= ua["time"]
        u["type"] -= ua["type"]
        ga, gb = _gaps(A[s]["dir"], unseen), _gaps(B[s]["dir"], unseen)
        u["sil"] = 0.0
        if ga is not None and gb is not None and len(ga) and len(gb):
            g = ga.merge(gb, on="seq", suffixes=("_a", "_b"))
            d = (g.nll_sil_b - g.nll_sil_a).groupby(g.seq).sum()
            u.loc[u.index.intersection(d.index), "sil"] = d.reindex(u.index.intersection(d.index)).values
        per_seed.append(u)
    if not per_seed:
        return None
    allu = pd.concat(per_seed).groupby(level=0).mean()
    return allu.reset_index()


def boot(u: pd.DataFrame, col, n_boot=1000, seed=0):
    """Gain per event and 95% interval from a bootstrap over users."""
    if u is None or len(u) == 0:
        return np.nan, np.nan
    v = u[col].values if col != "total" else (u.time + u.type + u.sil).values
    n = u.n.values
    est = v.sum() / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (n_boot, len(u)))
    bs = v[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(est), float((hi - lo) / 2)


def table_conditions(runs, out, R):
    rs3 = pick(runs, experiment="rq3")
    if not rs3:
        return
    data = {"nll": {}, "comparisons": {}}
    lines1 = []
    for cond, name in COND_NAMES.items():
        cells = []
        for b in ("event", "token"):
            nll = metric(pick(rs3, condition=cond, budget=b, variant=cond), "nll")
            data["nll"][f"{cond}/{b}"] = nll
            cells.append(fmt(nll[0], nll[1]))
        lines1.append(f"{name} & " + " & ".join(cells) + "\\\\")
    lines2 = []
    for a_, b_, label, meaning in COMPARISONS:
        cells = []
        for budget in ("event", "token"):
            A = _by_seed(pick(rs3, condition=a_, budget=budget, variant=a_))
            B = _by_seed(pick(rs3, condition=b_, budget=budget, variant=b_))
            u = paired_users(A, B)
            parts = {"nll": boot(u, "total"), "nll_time": boot(u, "time"), "nll_type": boot(u, "type"),
                     "nll_sil": boot(u, "sil")}
            data["comparisons"][f"{a_}-{b_}/{budget}"] = parts
            cells += [fmt(parts["nll"][0], parts["nll"][1], sign=True), fmt(parts["nll_time"][0], sign=True),
                      fmt(parts["nll_type"][0], sign=True), fmt(parts["nll_sil"][0], sign=True)]
        lines2.append(f"{label} & {meaning} & " + " & ".join(cells) + "\\\\")
    tex1 = ("\\begin{table}[t]\n\\caption{E3 on Amazon reviews: held-out NLL of each input (nats per event; "
            "time, type and silence parts together) under the event budget and the token budget. Means over seeds.}\n"
            "\\label{tab:rq3inputs}\n\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
            "Input & Event budget & Token budget\\\\\n\\midrule\n" + "\n".join(lines1) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(out, "rq3_inputs", tex1, R, "rq3_inputs", data["nll"])
    tex = ("\\begin{table*}[t]\n\\caption{E3 on Amazon reviews: gain of the first input over the second (reduction in NLL, "
           "nats per event; positive means the first input helps), split exactly into time, type, and silence parts "
           "(Eq.~\\ref{eq:dnll}), under the event budget and the token budget. Gains are averaged over seeds; $\\pm$ gives the "
           "half-width of a 95\\% interval from a paired bootstrap over users.}\n\\label{tab:rq3cond}\n"
           "\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}llcccccccc@{}}\n\\toprule\n & & \\multicolumn{4}{c}{Event budget} & \\multicolumn{4}{c}{Token budget}\\\\\n"
           "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\nComparison & Isolates & Total & Time & Type & Silence & Total & Time & Type & Silence\\\\\n"
           "\\midrule\n" + "\n".join(lines2) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write(out, "rq3_conditions", tex, R, "rq3_conditions", data)


def _events(run_dir, unseen=False):
    p = Path(run_dir) / ("test_unseen_events.csv.gz" if unseen else "test_events.csv.gz")
    return pd.read_csv(p) if p.exists() else None


def _paired_events(A: dict, B: dict, unseen=False):
    """Per-event gains of A over B averaged over seeds, with the event's features."""
    per = []
    for s in sorted(set(A) & set(B)):
        a, b = _events(A[s]["dir"], unseen), _events(B[s]["dir"], unseen)
        if a is None or b is None:
            continue
        m = a.merge(b, on=["seq", "idx"], suffixes=("", "_b"))
        m["time"] = m.nll_time_b - m.nll_time
        m["type"] = m.nll_type_b - m.nll_type
        per.append(m)
    if not per:
        return None
    m = pd.concat(per)
    feats = [c for c in ("k", "prev_k", "hist_len", "sim_prev", "prev_len") if c in m]
    g = m.groupby(["seq", "idx"])
    out = g[["time", "type"]].mean().join(g[feats].first()).reset_index()
    return out


def _stratum_rows(ev: pd.DataFrame, col: str, keys: list):
    rows = {}
    for k in keys:
        d = ev[ev[col] == k]
        if len(d) == 0:
            continue
        u = d.groupby("seq").agg(n=("idx", "size"), time=("time", "sum"), type=("type", "sum")).reset_index()
        u["sil"] = 0.0
        rows[k] = {"n": int(len(d)), "time": boot(u, "time"), "type": boot(u, "type"), "total": boot(u, "total")}
    return rows


def table_strata(runs, out, R, figs: Path):
    rs3 = pick(runs, experiment="rq3")
    A = _by_seed(pick(rs3, condition="full", budget="event", variant="full"))
    B = _by_seed(pick(rs3, condition="no_text", budget="event", variant="no_text"))
    ev = _paired_events(A, B)
    if ev is None:
        return
    cfg = next(iter(A.values()))["config"]
    meta = load_json(data_dir(cfg) / "meta.json")
    K = int(meta["K"])
    cnt = np.zeros(K)
    for r in read_jsonl(data_dir(cfg) / "train.jsonl"):
        cnt += np.bincount(r["k"], minlength=K)
    order = np.argsort(cnt)
    freq_group = np.empty(K, dtype=object)
    for i, t in enumerate(order):
        freq_group[t] = "rare" if i < K / 3 else ("medium" if i < 2 * K / 3 else "frequent")
    parent = np.array(meta.get("type_parent", list(range(K))))
    ev["hist"] = pd.cut(ev.hist_len, [0, 1, 4, 16, 10 ** 9], labels=["1", "2--4", "5--16", "17+"]).astype(str)
    ev["freq"] = [freq_group[k] for k in ev.k]
    ev["rel"] = np.where(ev.k == ev.prev_k, "same type",
                         np.where(parent[ev.k] == parent[ev.prev_k], "same parent", "other"))
    groups = [("hist", "Earlier events", ["1", "2--4", "5--16", "17+"]),
              ("freq", "Type frequency in training", ["rare", "medium", "frequent"]),
              ("rel", "Category of the previous event", ["same type", "same parent", "other"])]
    if "sim_prev" in ev and ev.sim_prev.notna().any():
        edges = np.unique([-np.inf, *ev.sim_prev.quantile([0.25, 0.5, 0.75]).values, np.inf])
        if len(edges) == 5:
            ev["sim"] = pd.cut(ev.sim_prev, edges, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"]).astype(str)
            groups.append(("sim", "Similarity of consecutive reviews", ["Q1 (least)", "Q2", "Q3", "Q4 (most)"]))
    if "prev_len" in ev and ev.prev_len.notna().any():
        edges = np.unique([-np.inf, *ev.prev_len.quantile([1 / 3, 2 / 3]).values, np.inf])
        if len(edges) == 4:   # skipped when lengths are too uniform to form three groups
            ev["plen"] = pd.cut(ev.prev_len, edges, labels=["short", "medium", "long"]).astype(str)
            groups.append(("plen", "Length of the previous review", ["short", "medium", "long"]))
    lines, data = [], {}
    for col, title, keys in groups:
        rows = _stratum_rows(ev, col, keys)
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\")
        for k, r in rows.items():
            data[f"{title}/{k}"] = r
            lines.append(f"\\quad {k} & {r['n']} & {fmt(*r['time'], sign=True)} & {fmt(*r['type'], sign=True)} & "
                         f"{fmt(*r['total'], sign=True)}\\\\")
    # cold start: users never seen in training
    u = paired_users(A, B, unseen=True)
    if u is not None and len(u):
        lines.append("\\multicolumn{5}{@{}l}{\\emph{Users not seen in training}}\\\\")
        r = {"n": int(u.n.sum()), "time": boot(u, "time"), "type": boot(u, "type"), "total": boot(u, "total")}
        data["unseen users"] = r
        lines.append(f"\\quad all & {r['n']} & {fmt(*r['time'], sign=True)} & {fmt(*r['type'], sign=True)} & "
                     f"{fmt(*r['total'], sign=True)}\\\\")
    # training size: FULL vs NO-TEXT trained on a fraction of the training users
    for frac in (0.1, 0.3):
        Af = _by_seed(pick(rs3, variant=f"full_frac{frac}"))
        Bf = _by_seed(pick(rs3, variant=f"no_text_frac{frac}"))
        uf = paired_users(Af, Bf)
        if uf is None or not len(uf):
            continue
        if not any("training users" in x for x in lines):
            lines.append("\\multicolumn{5}{@{}l}{\\emph{Share of training users}}\\\\")
        r = {"n": int(uf.n.sum()), "time": boot(uf, "time"), "type": boot(uf, "type"), "total": boot(uf, "total")}
        data[f"train_frac/{frac}"] = r
        lines.append(f"\\quad {int(frac * 100)}\\% & {r['n']} & {fmt(*r['time'], sign=True)} & {fmt(*r['type'], sign=True)} & "
                     f"{fmt(*r['total'], sign=True)}\\\\")
    tex = ("\\begin{table}[t]\n\\caption{Where the reviews help: gain of FULL over NO-TEXT (event budget; nats per event, "
           "positive means the reviews help), split into time and type parts, by stratum. $\\pm$: half-width of a 95\\% "
           "interval from a paired bootstrap over users; gains are averaged over seeds.}\n\\label{tab:rq3strata}\n"
           "\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lrccc@{}}\n\\toprule\nStratum & Events & Time & Type & Total\\\\\n"
           "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(out, "rq3_strata", tex, R, "rq3_strata", data)
    try:   # figure: gain vs number of earlier events
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = _stratum_rows(ev, "hist", ["1", "2--4", "5--16", "17+"])
        ks = list(rows)
        fig, ax = plt.subplots(figsize=(3.3, 2.0))
        for part, mk, col in (("time", "o-", "#2F5D8A"), ("type", "s-", "#E9A23B")):
            y = np.array([rows[k][part][0] for k in ks])
            e = np.array([rows[k][part][1] for k in ks])
            ax.errorbar(range(len(ks)), y, yerr=e, fmt=mk, color=col, label=part, capsize=2, lw=1.2, ms=3.5)
        ax.axhline(0, color="#6C757D", lw=0.6)
        ax.set_xticks(range(len(ks)), [k.replace("--", "-") for k in ks])
        ax.set_xlabel("earlier events of the user")
        ax.set_ylabel("gain (nats/event)")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        figs.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs / "rq3_history.pdf")
        print(f"wrote {figs / 'rq3_history.pdf'}")
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


def table_named_gen(runs, out, R):
    rs3 = pick(runs, experiment="rq3")
    lines, data = [], {}
    for var, label in (("no_text", "NO-TEXT"), ("named_out_no_text", "NO-TEXT, NAMED-OUT"),
                       ("full", "FULL"), ("named_out_full", "FULL, NAMED-OUT")):
        rs = pick(rs3, variant=var, budget="event")
        if not rs:
            continue
        nll, ty, f1 = (metric(rs, k) for k in ("nll", "nll_type", "type_macro_f1"))
        data[var] = {"nll": nll, "nll_type": ty, "macro_f1": f1}
        lines.append(f"{label} & {fmt(nll[0], nll[1])} & {fmt(ty[0], ty[1])} & {fmt(f1[0], f1[1])}\\\\")
    if lines:
        tex = ("\\begin{table}[t]\n\\caption{Type names on the output side (NAMED-OUT): the type tokens start from the "
               "LLM's embedding of the type name. NLL, its type part (nats per event), and type macro-F1.}\n"
               "\\label{tab:rq3named}\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
               "Input & NLL & Type NLL & Macro-F1\\\\\n\\midrule\n" + "\n".join(lines) +
               "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        write(out, "rq3_named", tex, R, "rq3_named", data)
    lines, data = [], {}
    for var, lam in (("full", 0.0), ("gen0.1", 0.1), ("gen1.0", 1.0)):
        rs = pick(rs3, variant=var, budget="event")
        if not rs:
            continue
        nt, nk, nll = (metric(rs, k) for k in ("nll_time", "nll_type", "nll"))
        ppl = metric(rs, "desc_ppl")
        rl = ms([r.get("generation", {}).get("rouge_l") for r in rs])
        ov = ms([r.get("generation", {}).get("ngram8_overlap") for r in rs])
        data[var] = {"nll_time": nt, "nll_type": nk, "nll": nll, "desc_ppl": ppl, "rouge_l": rl, "overlap8": ov}
        lines.append(f"{lam:g} & {fmt(nt[0], nt[1])} & {fmt(nk[0], nk[1])} & {fmt(nll[0], nll[1])} & {fmt(ppl[0], d=1)} & "
                     f"{fmt(100 * rl[0] if np.isfinite(rl[0]) else np.nan, d=1)} & {fmt(100 * ov[0] if np.isfinite(ov[0]) else np.nan, d=1)}\\\\")
    if lines:
        tex = ("\\begin{table}[t]\n\\caption{Writing the next description (FULL+GEN) with description-loss weight "
               "$\\lambda_{\\text{d}}$ ($0$ is FULL). Time, type and total NLL (nats per event), description "
               "perplexity, ROUGE-L of sampled reviews (\\%), and share of generated 8-grams found in the training "
               "reviews (\\%).}\n\\label{tab:rq3gen}\n\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}ccccccc@{}}\n"
               "\\toprule\n$\\lambda_{\\text{d}}$ & Time & Type & NLL & PPL & ROUGE-L & 8-gram\\\\\n\\midrule\n"
               + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        write(out, "rq3_gen", tex, R, "rq3_gen", data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--data", default=None, help="folder that contains the data sets (default: ./data); "
                    "needed when the runs were made on another computer")
    a = ap.parse_args()
    global DATA_ROOT
    DATA_ROOT = Path(a.data) if a.data else None
    runs = load_runs(Path(a.runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} finished runs")
    R = {}
    table_hawkes(runs, out, R)
    table_bench(runs, out, R)
    table_silence(runs, out, R)
    table_conditions(runs, out, R)
    table_strata(runs, out, R, out)
    table_named_gen(runs, out, R)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
