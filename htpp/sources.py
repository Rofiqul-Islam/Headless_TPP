"""Builders for the three data sources used in the paper:

* synthetic multivariate Hawkes sequences with a known likelihood (RQ1),
* the standard EasyTPP benchmarks (Retweet, StackOverflow, Taobao, Amazon, Taxi) (RQ1),
* Amazon Reviews 2023 with review text, one sequence per user (RQ3).
"""
from __future__ import annotations

import gzip
import io
import json
import pickle
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import hawkes
from .data import EventSeq, censor_random_cut, split_long


# ----------------------------------------------------------------------------
# synthetic Hawkes
# ----------------------------------------------------------------------------

def build_hawkes(n_train=2000, n_val=300, n_test=500, K=3, T=60.0, seed=0, branching_ratio=0.6,
                 mean_rate=1.0, min_events=3):
    rng = np.random.default_rng(seed)
    mu, alpha, beta = hawkes.default_params(K, rng, branching_ratio, mean_rate)
    splits = {}
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        seqs = []
        i = 0
        while len(seqs) < n:
            t, k = hawkes.simulate(mu, alpha, beta, T, rng)
            i += 1
            if len(t) < min_events:
                continue
            ll = hawkes.loglik(mu, alpha, beta, t, k, T, t0=t[0])  # scored part: events 2..N + final gap
            seqs.append(EventSeq(f"{name}{i}", t, k, t_end=float(T), true_nll=float(-ll)))
        splits[name] = seqs
    meta = {"source": "hawkes", "K": K, "type_names": [f"type {j}" for j in range(K)],
            "time_unit": "unit", "time_resolution": 1e-6, "has_text": False,
            "hawkes": {"mu": mu.tolist(), "alpha": alpha.tolist(), "beta": beta, "T": T}}
    return splits, meta


# ----------------------------------------------------------------------------
# EasyTPP benchmarks
# ----------------------------------------------------------------------------

_EASYTPP_SPLITS = {"train": ["train"], "val": ["validation", "valid", "dev"], "test": ["test"]}


def _easytpp_rows_to_seqs(rows, prefix):
    seqs = []
    for i, r in enumerate(rows):
        if isinstance(r, dict) and "time_since_start" in r and isinstance(r["time_since_start"], list):
            t = np.asarray(r["time_since_start"], float)
            k = np.asarray(r["type_event"], int)
        else:  # pickle format: list of event dicts
            t = np.asarray([e["time_since_start"] for e in r], float)
            k = np.asarray([e["type_event"] for e in r], int)
        order = np.argsort(t, kind="stable")
        t, k = t[order], k[order]
        if len(t) >= 2:
            seqs.append(EventSeq(f"{prefix}{i}", t, k))
    return seqs


def load_easytpp(name: str, local_dir: str | None = None):
    """Return ({train,val,test: [EventSeq]}, K). Reads the Hugging Face copies
    (`easytpp/<name>`) or a local folder with EasyTPP pickle/json files."""
    raw = {}
    K = None
    if local_dir:
        d = Path(local_dir)
        for split, cands in _EASYTPP_SPLITS.items():
            for c in cands:
                for ext in ("pkl", "json", "jsonl"):
                    p = d / f"{c}.{ext}"
                    if not p.exists():
                        continue
                    if ext == "pkl":
                        with open(p, "rb") as f:
                            obj = pickle.load(f, encoding="latin1")
                        K = int(obj.get("dim_process", K or 0))
                        key = [x for x in obj if x not in ("dim_process",)][0]
                        raw[split] = obj[key]
                    elif ext == "json":
                        with open(p, encoding="utf-8") as f:
                            obj = json.load(f)
                        raw[split] = obj if isinstance(obj, list) else obj.get("data", obj)
                    else:
                        with open(p, encoding="utf-8") as f:
                            raw[split] = [json.loads(x) for x in f if x.strip()]
                    break
                if split in raw:
                    break
    else:
        from datasets import load_dataset  # imported lazily
        ds = load_dataset(f"easytpp/{name}")
        for split, cands in _EASYTPP_SPLITS.items():
            for c in cands:
                if c in ds:
                    raw[split] = [dict(r) for r in ds[c]]
                    break
    missing = [s for s in ("train", "val", "test") if s not in raw]
    if missing:
        raise FileNotFoundError(f"could not find EasyTPP splits {missing} for {name!r}")
    splits = {s: _easytpp_rows_to_seqs(raw[s], f"{s}") for s in raw}
    if K is None:
        first = raw["train"][0]
        if isinstance(first, dict) and "dim_process" in first:
            K = int(first["dim_process"])
        else:
            K = int(max(int(s.k.max()) for s in splits["train"]) + 1)
    return splits, K


def build_easytpp(name, local_dir=None, censor="none", split_long_at=None, seed=0, resolution=None):
    splits, K = load_easytpp(name, local_dir)
    rng = np.random.default_rng(seed)
    for s in list(splits):
        seqs = splits[s]
        if split_long_at:
            seqs = split_long(seqs, int(split_long_at))
        if censor == "random_cut":
            seqs = censor_random_cut(seqs, rng)
        splits[s] = seqs
    if resolution is None:
        taus = np.concatenate([q.tau[1:] for q in splits["train"]])
        pos = np.sort(np.unique(taus[taus > 0]))
        resolution = float(np.min(np.diff(pos))) if len(pos) > 2 else 1e-6
        resolution = max(resolution, 1e-6 * float(np.median(taus[taus > 0])))
    meta = {"source": f"easytpp/{name}", "K": int(K), "type_names": [f"type {j}" for j in range(K)],
            "time_unit": "dataset unit", "time_resolution": float(resolution), "has_text": False,
            "censor": censor}
    return splits, meta


# ----------------------------------------------------------------------------
# Amazon Reviews 2023 with text
# ----------------------------------------------------------------------------

def _open_any(path):
    path = str(path)
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


_USER_RE = re.compile(r'"(?:user_id|reviewerID)"\s*:\s*"([^"]*)"')


def _iter_jsonl(paths):
    for p in paths:
        with _open_any(p) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _scramble(name: str, rng: random.Random) -> str:
    """A fixed meaningless string with about the same length as the name."""
    cons, vow = "bcdfghjklmnprstvz", "aeiou"
    words = []
    for w in name.split():
        n = max(2, len(w))
        words.append("".join((rng.choice(cons) if i % 2 == 0 else rng.choice(vow)) for i in range(n)))
    return " ".join(words) if words else "zuvo"


def build_amazon(review_paths, meta_paths, cat_level=2, K=24, min_events=5, max_events_per_user=200,
                 max_users=20000, desc_max_chars=400, seed=0, split="chrono", train_end_q=0.8, val_end_q=0.9,
                 unseen_frac=0.1, val_frac=0.1, test_frac=0.1, time_unit_seconds=86400.0, t_end=None):
    """One sequence per user. Events are reviews (time = review timestamp), the type is
    the product category at depth `cat_level` of the item's category path (the K-1 most
    frequent categories plus "Other"), and the description is the review title and text.
    Works with the 2023 release (Hou et al.) and the 2018 release (Ni et al.).

    split="chrono" (the protocol of Section 4): a group of `unseen_frac` users is held
    out completely (split test_unseen); for the other users, events before the
    `train_end_q` time quantile are training, events up to `val_end_q` are validation and
    later events are test. Validation and test sequences keep the user's earlier events as
    history only (field n_hist), so a user's past informs predictions but is never scored
    twice. Each sequence ends at the end of its period, so its final quiet gap is genuinely
    censored. split="user": plain split of users into train/val/test."""
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    # The review files can hold tens of millions of reviews, so they are streamed three
    # times instead of being loaded: (1) count reviews per user, (2) keep the reviews of a
    # random sample of active users, (3) read categories only for the items they reviewed.
    cnt_u = Counter()
    for p in review_paths:
        with _open_any(p) as f:
            for line in f:
                m = _USER_RE.search(line)
                if m:
                    cnt_u[m.group(1)] += 1
    cand = sorted(u for u, c in cnt_u.items() if c >= min_events)
    del cnt_u
    pyrng.shuffle(cand)
    keep = set(cand[: int(max_users * 1.5)] if max_users else cand)   # spare users for those lost to missing metadata
    raw_rev = defaultdict(list)
    for p in review_paths:
        with _open_any(p) as f:
            for line in f:
                m = _USER_RE.search(line)
                if not m or m.group(1) not in keep:
                    continue
                r = json.loads(line)
                ts = r.get("timestamp") or r.get("unixReviewTime")
                if ts is None:
                    continue
                ts = float(ts)
                if ts > 1e11:      # milliseconds
                    ts /= 1000.0
                title = (r.get("title") or r.get("summary") or "").strip()
                body = (r.get("text") or r.get("reviewText") or "").strip()
                text = (title + ". " + body).strip(". ").strip()
                raw_rev[m.group(1)].append((ts, r.get("parent_asin") or r.get("asin"), text[:desc_max_chars]))
    items = {it for ev in raw_rev.values() for _, it, _ in ev}
    item_cat = {}
    for line_obj in _iter_jsonl(meta_paths):
        m = line_obj
        it = m.get("parent_asin") or m.get("asin")
        if it not in items:
            continue
        cats = m.get("categories") or m.get("category") or []
        if cats and isinstance(cats[0], list):      # some releases nest the path
            cats = cats[0]
        if not cats:
            mc = m.get("main_category") or m.get("main_cat")
            cats = [mc] if mc else []
        if not cats:
            continue
        lvl = min(cat_level, len(cats) - 1)
        item_cat[it] = (cats[lvl], cats[lvl - 1] if lvl >= 1 else cats[0])
    by_user = {u: [e for e in ev if e[1] in item_cat] for u, ev in raw_rev.items()}
    del raw_rev
    users = [u for u in cand if u in by_user and len(by_user[u]) >= min_events]
    users = users[: int(max_users)] if max_users else users
    # 3) categories -> types (top K-1 + Other)
    cnt = Counter()
    for u in users:
        for _, item, _ in by_user[u]:
            cnt[item_cat[item][0]] += 1
    top = [c for c, _ in cnt.most_common(K - 1)]
    type_of = {c: j for j, c in enumerate(top)}
    names = top + ["Other"]
    parent_of = {}
    for item, (leaf, parent) in item_cat.items():
        if leaf in type_of:
            parent_of[type_of[leaf]] = parent
    parents = sorted(set(parent_of.values())) + ["Other"]
    type_parent = [parents.index(parent_of[j]) if j in parent_of else len(parents) - 1 for j in range(K)]
    # 4) sequences (sorted, exact duplicates removed, equal timestamps broken by 1 s steps)
    seqs = []
    for u in users:
        ev = sorted(set(by_user[u]), key=lambda x: (x[0], x[1]))[-max_events_per_user:]
        ts = np.array([e[0] for e in ev])
        for i in range(1, len(ts)):
            if ts[i] <= ts[i - 1]:
                ts[i] = ts[i - 1] + 1.0
        k = [type_of.get(item_cat[e[1]][0], K - 1) for e in ev]
        seqs.append(dict(user=u, ts=ts, k=np.array(k), text=[e[2] for e in ev]))
    all_ts = np.concatenate([s["ts"] for s in seqs])
    end = float(t_end) if t_end is not None else float(all_ts.max())
    t0 = float(all_ts.min())
    # 5) splits
    n = len(seqs)
    idx = rng.permutation(n)
    parts = {}
    if split == "chrono":
        T1, T2 = (float(np.quantile(all_ts, q)) for q in (train_end_q, val_end_q))
        n_un = int(round(unseen_frac * n))
        unseen, seen = idx[:n_un], idx[n_un:]
        parts["test_unseen"] = [(i, 0, len(seqs[i]["ts"]), 0, end) for i in unseen]
        for name, lo, hi in (("train", -np.inf, T1), ("val", T1, T2), ("test", T2, np.inf)):
            rows = []
            for i in seen:
                ts = seqs[i]["ts"]
                a = int(np.searchsorted(ts, lo, side="left")) if np.isfinite(lo) else 0
                b = int(np.searchsorted(ts, hi, side="left")) if np.isfinite(hi) else len(ts)
                if name == "train":
                    if b >= 2:
                        rows.append((i, 0, b, 0, T1))
                elif b > a:
                    rows.append((i, 0, b, a, T2 if name == "val" else end))   # history 0..a-1, scored a..b-1
            parts[name] = rows
        meta_split = {"split": "chrono", "train_end_unix": T1, "val_end_unix": T2, "unseen_users": int(n_un)}
    else:
        n_test, n_val = int(round(test_frac * n)), int(round(val_frac * n))
        groups = {"test": idx[:n_test], "val": idx[n_test:n_test + n_val], "train": idx[n_test + n_val:]}
        parts = {g: [(i, 0, len(seqs[i]["ts"]), 0, end) for i in ids] for g, ids in groups.items()}
        meta_split = {"split": "user"}
    # 6) SHUFFLED text: another user's earlier review of the same type (from the training pool)
    pool = defaultdict(list)
    for (i, a, b, h, te) in parts["train"]:
        s = seqs[i]
        for t, k, x in zip(s["ts"][a:b], s["k"][a:b], s["text"][a:b]):
            pool[int(k)].append((t, i, x))
    for k in pool:
        pool[k].sort()
    pool_t = {k: np.array([p[0] for p in v]) for k, v in pool.items()}

    def shuffled(i, ts, ks):
        out = []
        for t, k in zip(ts, ks):
            cands = pool.get(int(k), [])
            if not cands:
                out.append("")
                continue
            hi = int(np.searchsorted(pool_t[int(k)], t, side="left"))
            choice = None
            for _ in range(8):
                j = pyrng.randrange(hi) if hi > 0 else pyrng.randrange(len(cands))
                if cands[j][1] != i:
                    choice = cands[j][2]
                    break
            out.append(choice if choice is not None else cands[pyrng.randrange(len(cands))][2])
        return out

    splits = {}
    for name, rows in parts.items():
        out = []
        for (i, a, b, h, te) in rows:
            s = seqs[i]
            ts = s["ts"][a:b]
            out.append(EventSeq(str(s["user"]), (ts - t0) / time_unit_seconds, s["k"][a:b], text=s["text"][a:b],
                                shuf=shuffled(i, ts, s["k"][a:b]), t_end=float((max(te, ts[-1]) - t0) / time_unit_seconds),
                                n_hist=int(h)))
        splits[name] = out
    scr = [_scramble(nm, pyrng) for nm in names]
    meta = {"source": "amazon-reviews", "K": K, "type_names": names, "scrambled_names": scr,
            "type_parent": type_parent, "parent_names": parents, "time_unit": "days",
            "time_resolution": 1.0 / time_unit_seconds, "has_text": True, "cat_level": cat_level,
            "origin_unix": t0, "end_unix": end, **meta_split}
    return splits, meta
