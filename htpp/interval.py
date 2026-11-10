"""Likelihood of the recorded interval: a fair score when timestamps are rounded.

When logs store rounded times (whole seconds, say), the recorded waiting time is only
known up to the rounding step delta, and a continuous density evaluated at the rounded
value rewards a model for putting sharp peaks on the recorded values. Every model is
therefore also scored by the probability it gives to the interval of waiting times that
is consistent with the record,

    cell(tau) = [tau, tau + delta),

reported as -log(P(cell) / width(cell)) so that it stays in nats per event and equals
the density NLL for a smooth density. All models are scored on the same cells.
"""
from __future__ import annotations

import math

import numpy as np


def _round_sig(x: np.ndarray, sig: int = 3) -> np.ndarray:
    x = np.asarray(x, float)
    e = np.floor(np.log10(np.abs(x)))
    f = 10.0 ** (e - sig + 1)
    return np.round(x / f) * f


def estimate_grid(taus, max_n: int = 500_000):
    """Return (grid step or None, info). A grid is reported when at least 90% of the
    waiting times are multiples of the most common spacing between distinct values."""
    t = np.asarray(taus, float)
    t = t[np.isfinite(t) & (t >= 0)]
    info = {"n": int(len(t)), "zero_share": float(np.mean(t == 0)) if len(t) else 0.0}
    if len(t) > max_n:
        t = np.random.default_rng(0).choice(t, max_n, replace=False)
    pos = np.unique(t[t > 0])
    if len(pos) < 20:
        return None, info
    d = np.diff(pos)
    d = d[d > 1e-9 * pos.max()]
    if not len(d):
        return None, info
    vals, cnt = np.unique(_round_sig(d), return_counts=True)
    q = float(vals[np.argmax(cnt)])
    on = float(np.mean(np.abs(t / q - np.round(t / q)) < 1e-3))
    info.update({"grid_candidate": q, "on_grid": on})
    return (q if on >= 0.9 else None), info


def choose_delta(taus):
    """The recording step if the waiting times lie on a grid, else 1/1000 of the median
    positive waiting time (one significant digit)."""
    taus = np.asarray(taus, float)
    q, info = estimate_grid(taus)
    if q is None:
        med = float(np.median(taus[taus > 0]))
        q = float(_round_sig(np.array([1e-3 * med]), 1)[0])
        info["fallback"] = True
    info["delta"] = q
    return q, info


def cells(tau, delta: float):
    """The recorded interval [tau, tau + delta): with timestamps stored at step delta,
    a recorded waiting time tau stands for the true waits from tau up to tau + delta."""
    tau = np.asarray(tau, float)
    return tau, tau + delta


def headless_interval_logp(tc, lvl_lp: list[np.ndarray], path: np.ndarray, plen: np.ndarray,
                           lo: np.ndarray, hi: np.ndarray, branch: dict | None = None,
                           rows: np.ndarray | None = None):
    """log P(lo <= tau < hi) for each event under Headless-TPP.

    lvl_lp[l][i]: log-probabilities over the children at level l, read along event i's own
    path (level 0 includes the tail). branch[i]: log-probabilities from the expanded token
    tree, one array per level l < depth with one row per prefix (as in TimeCode.fine_probs).
    rows[i] maps output row i to its index in lvl_lp (default: the same index).
    A part of the interval whose distribution is unknown (a node off the event's path and
    deeper than the expanded tree) gets its node's probability spread by width; such events
    are flagged in `approx`. With the full tree expanded the result is exact.
    """
    E, W, S, B, L = tc.edges, tc.widths, tc._strides, tc.branching, tc.L
    n = len(lo)
    rows = np.arange(n) if rows is None else rows
    out = np.empty(n)
    approx = np.zeros(n, dtype=bool)
    beta, a_max = float(tc.beta), float(tc.a_max)
    for o in range(n):
        i = int(rows[o])
        br = branch.get(i) if branch else None
        own = None if plen[i] < L and path[i, 0] == tc.tail_index else int(sum(path[i, l] * S[l] for l in range(L)))
        a, b = float(lo[o]), float(hi[o])
        tot = 0.0
        bf = min(b, a_max)
        if bf > a:
            j0 = max(int(np.searchsorted(E, a, side="right")) - 1, 0)
            j1 = min(int(np.searchsorted(E, bf, side="left")) - 1, tc.n_fine - 1)
            for j in range(j0, j1 + 1):
                ov = min(bf, E[j + 1]) - max(a, E[j])
                if ov <= 0:
                    continue
                p = 1.0
                for lv in range(L):
                    node = j // S[lv - 1] if lv > 0 else 0          # index of the prefix (level lv-1 node)
                    if br is not None and lv < len(br):
                        vec = br[lv][node]
                    elif lv < plen[i] and own is not None and (lv == 0 or node == own // S[lv - 1]):
                        vec = lvl_lp[lv][i]
                    elif lv == 0:
                        vec = lvl_lp[0][i]
                    else:                                          # unknown: spread the node by width
                        st = node * S[lv - 1]
                        p *= W[j] / (E[st + S[lv - 1]] - E[st])
                        approx[o] = True
                        break
                    p *= math.exp(vec[(j // S[lv]) % B[lv]])
                tot += p * ov / W[j]
        if b > a_max:
            vec0 = br[0][0] if br is not None else lvl_lp[0][i]
            p_tail = math.exp(vec0[tc.tail_index])
            a2 = max(a, a_max)
            tot += p_tail * (math.exp(-beta * (a2 - a_max)) - math.exp(-beta * (b - a_max)))
        out[o] = math.log(max(tot, 1e-300))
    return out, approx


def interval_nll(logp: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """-log(P(cell) / width(cell)) per event (nats; comparable with a density NLL)."""
    return -(logp - np.log(hi - lo))
