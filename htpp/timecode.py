"""Hierarchical time code (Section 3 of the paper).

A waiting time tau >= 0 is written as a path of time tokens (c_1, ..., c_L):
level 1 chooses one of b_1 coarse bins or the tail [a_max, inf); each later level
chooses one of b_l children of the bin above. The leaves ("fine bins") partition
[0, a_max). Probability inside a fine bin is spread evenly (u_B = 1/w); in the tail
it decays exponentially with a fixed rate beta. With these rules the LLM's
probabilities over the time tokens define an exact density and survival function:

    log f(tau) = sum_l log q_l(c_l) + log u_B(tau)
    S(tau)     = sum_l (prod_{j<l} q_j) Q_l + (prod_l q_l) r_B(tau)

where q_l is the probability of the token on tau's path at level l, Q_l the total
probability of the later tokens at that level, and r_B(tau) the share of tau's bin
that lies after tau. `kind="digits"` gives the LLMTime-style ablation: fixed-point
decimal digits (branching 10 at every level, equal-width bins) written with the base
tokenizer's digit tokens.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

NEG = -1.0e4  # finite stand-in for log(0); keeps gradients free of NaN


@dataclass
class TimeCode:
    branching: list[int]
    edges: np.ndarray          # fine-bin edges: len prod(branching)+1, edges[0]=0, edges[-1]=a_max
    beta: float                # tail rate
    kind: str = "hier"         # "hier" or "digits"
    resolution: float = 0.0

    # ------------------------------------------------------------------ basics
    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.float64)
        if len(self.edges) != self.n_fine + 1:
            raise ValueError("edges must have prod(branching)+1 entries")
        if not np.all(np.diff(self.edges) > 0):
            raise ValueError("edges must be strictly increasing")
        self.widths = np.diff(self.edges)
        st, p = [], 1
        for b in reversed(self.branching):
            st.append(p)
            p *= b
        self._strides = list(reversed(st))

    @property
    def L(self) -> int:
        return len(self.branching)

    @property
    def a_max(self) -> float:
        return float(self.edges[-1])

    @property
    def n_fine(self) -> int:
        return int(np.prod(self.branching))

    @property
    def tail_index(self) -> int:
        return self.branching[0]

    def level_size(self, level: int) -> int:
        """Number of tokens allowed at 0-based `level` (level 0 includes the tail)."""
        return self.branching[level] + (1 if level == 0 else 0)

    def to_dict(self) -> dict:
        return {"branching": list(self.branching), "edges": self.edges.tolist(), "beta": self.beta,
                "kind": self.kind, "resolution": self.resolution}

    @classmethod
    def from_dict(cls, d: dict) -> "TimeCode":
        return cls(branching=list(d["branching"]), edges=np.asarray(d["edges"]), beta=float(d["beta"]),
                   kind=d.get("kind", "hier"), resolution=float(d.get("resolution", 0.0)))

    # ----------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, taus, branching=(8, 8, 8), p_tail=0.01, resolution=None, split="quantile") -> "TimeCode":
        """Data-driven hierarchical bins. Level 1 uses quantiles of the training waiting
        times (so coarse bins carry similar mass); each bin is split recursively with
        the quantiles of the waiting times that fall in it, falling back to geometric
        (or, for a bin starting at 0, linear) splits when data are scarce. Bins are kept
        at least `resolution` wide (rule R3)."""
        taus = np.asarray(taus, dtype=np.float64)
        taus = taus[np.isfinite(taus) & (taus >= 0)]
        if len(taus) == 0:
            raise ValueError("no waiting times to fit the time code")
        pos = taus[taus > 0]
        if resolution is None or resolution <= 0:
            resolution = 1e-3 * float(np.median(pos)) if len(pos) else 1e-6
            warnings.warn(f"time resolution not given; using {resolution:.3g} (0.001 x median wait)")
        n_fine = int(np.prod(branching))
        a_max = float(np.quantile(taus, 1.0 - p_tail))
        a_max = max(a_max, resolution * n_fine * 1.0001)
        edges = _split_recursive(0.0, a_max, list(branching), taus[taus < a_max], resolution, split)
        tail = taus[taus >= a_max] - a_max
        beta = 1.0 / max(float(np.mean(tail)), resolution) if len(tail) >= 5 else 1.0 / a_max
        return cls(branching=list(branching), edges=edges, beta=beta, kind="hier", resolution=resolution)

    @classmethod
    def digits(cls, n_int=3, n_dec=2, scale=1.0, taus=None) -> "TimeCode":
        """Fixed-point decimal code: tau/scale written with n_int integer and n_dec
        decimal digits; values >= 10**n_int go to the tail."""
        L = n_int + n_dec
        grid = np.arange(10 ** L + 1, dtype=np.float64) * (10.0 ** -n_dec) * scale
        a_max = float(grid[-1])
        beta = 1.0 / a_max
        if taus is not None:
            t = np.asarray(taus, float)
            tail = t[t >= a_max] - a_max
            if len(tail) >= 5:
                beta = 1.0 / max(float(np.mean(tail)), 1e-12)
        return cls(branching=[10] * L, edges=grid, beta=beta, kind="digits", resolution=(10.0 ** -n_dec) * scale)

    # ---------------------------------------------------------------- encoding
    def encode(self, tau):
        """Return (paths [n, L] with -1 padding, path lengths [n], fine-bin index [n] or -1 for tail)."""
        tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, None)
        n = tau.shape[0]
        paths = np.full((n, self.L), -1, dtype=np.int64)
        plen = np.full(n, self.L, dtype=np.int64)
        tail = tau >= self.a_max
        m = np.clip(np.searchsorted(self.edges, tau, side="right") - 1, 0, self.n_fine - 1)
        for lv in range(self.L):
            paths[:, lv] = (m // self._strides[lv]) % self.branching[lv]
        paths[tail, 0] = self.tail_index
        paths[tail, 1:] = -1
        plen[tail] = 1
        fine = np.where(tail, -1, m)
        return paths, plen, fine

    def log_inbin(self, tau, fine=None):
        """log u_B(tau): -log w for a finite bin, log beta - beta (tau - a_max) in the tail."""
        tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, None)
        if fine is None:
            _, _, fine = self.encode(tau)
        out = np.empty_like(tau)
        fin = fine >= 0
        out[fin] = -np.log(self.widths[fine[fin]])
        out[~fin] = math.log(self.beta) - self.beta * (tau[~fin] - self.a_max)
        return out

    def log_after_share(self, tau, fine=None):
        """log r_B(tau): share of tau's bin that lies after tau."""
        tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, None)
        if fine is None:
            _, _, fine = self.encode(tau)
        out = np.empty_like(tau)
        fin = fine >= 0
        f = fine[fin]
        share = (self.edges[f + 1] - tau[fin]) / self.widths[f]
        out[fin] = np.log(np.clip(share, 1e-300, 1.0))
        out[~fin] = -self.beta * (tau[~fin] - self.a_max)
        return out

    def bin_mids(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def tokens_per_event(self, plen) -> float:
        return float(np.mean(plen)) + 1.0  # time tokens + one mark token

    # ------------------------------------------------ distributions from levels
    def fine_probs(self, level_logps: list[np.ndarray], depth: int | None = None):
        """Probabilities of all fine bins and the tail from the expanded tree.

        level_logps[l] has shape [prod(b_1..b_l), level_size(l)]: the log-probabilities
        at level l for every prefix (level 0: one row with the tail as last column).
        Levels >= depth are treated as uniform inside their parent (approximation used
        only for the deep digit tree). Returns (p_fine [n_fine], p_tail).
        """
        depth = self.L if depth is None else min(depth, self.L)
        p0 = np.exp(level_logps[0][0])
        p_tail = float(p0[self.tail_index])
        p = p0[: self.branching[0]]
        for lv in range(1, self.L):
            if lv < depth:
                child = np.exp(level_logps[lv])            # [prod, b_lv]
                p = (p[:, None] * child).reshape(-1)
            else:
                p = np.repeat(p / self.branching[lv], self.branching[lv])
        return p, p_tail

    def mean_median(self, p_fine: np.ndarray, p_tail: float, cap: float | None = None):
        """Mean (of min(tau, cap) when a cap is given) and median of the predictive distribution."""
        tot = p_fine.sum() + p_tail
        p_fine = p_fine / tot
        p_tail = p_tail / tot
        if cap is None or not np.isfinite(cap):
            mean = float(np.dot(p_fine, self.bin_mids()) + p_tail * (self.a_max + 1.0 / self.beta))
        else:
            lo, hi, w = self.edges[:-1], self.edges[1:], self.widths
            c = float(cap)
            inside = ((np.clip(c, lo, hi) - lo) * (np.clip(c, lo, hi) + lo) / 2 + c * (hi - np.clip(c, lo, hi))) / w
            m_bin = np.where(c >= hi, 0.5 * (lo + hi), np.where(c <= lo, c, inside))
            m_tail = c if c <= self.a_max else self.a_max + (1.0 - math.exp(-self.beta * (c - self.a_max))) / self.beta
            mean = float(np.dot(p_fine, m_bin) + p_tail * m_tail)
        cdf = np.cumsum(p_fine)
        if cdf[-1] >= 0.5:
            m = int(np.searchsorted(cdf, 0.5))
            prev = cdf[m - 1] if m > 0 else 0.0
            frac = (0.5 - prev) / max(p_fine[m], 1e-300)
            median = float(self.edges[m] + frac * self.widths[m])
        else:  # median in the tail: S_tail(x) = p_tail exp(-beta x) = 0.5 - ... solve
            rem = (0.5 - cdf[-1]) / max(p_tail, 1e-300)
            median = float(self.a_max - math.log(max(1.0 - rem, 1e-300)) / self.beta)
        return mean, median


def _split_recursive(a, b, branching, x, res, split):
    """Edges of all fine bins inside [a, b) for the remaining branching factors."""
    if not branching:
        return np.array([a, b])
    n = branching[0]
    # every child must leave room for its own children to be at least `res` wide
    e = _split(a, b, n, x, res * float(np.prod(branching[1:])) if len(branching) > 1 else res, split)
    pieces = []
    for j in range(n):
        lo, hi = e[j], e[j + 1]
        sub = x[(x >= lo) & (x < hi)]
        child = _split_recursive(lo, hi, branching[1:], sub, res, split)
        pieces.append(child[:-1])
    pieces.append(np.array([b]))
    return np.concatenate(pieces)


def _split(a, b, n, x, res, split):
    if split == "quantile" and len(x) >= 10 * n:
        q = np.quantile(x, np.linspace(0.0, 1.0, n + 1)[1:-1])
        e = [a]
        for v in q:
            e.append(max(float(v), e[-1] + res))
        e.append(b)
        e = np.asarray(e)
        d = np.diff(e)
        if np.all(d > 0) and np.all(d >= res * (1 - 1e-9)):
            return e
    e = None
    if split != "linear" and a > 0:
        g = np.geomspace(a, b, n + 1)
        if np.all(np.diff(g) >= res * (1 - 1e-9)):
            e = g
    if e is None:
        e = np.linspace(a, b, n + 1)
    if np.any(np.diff(e) < res * (1 - 1e-9)):
        warnings.warn(f"bin [{a:.4g},{b:.4g}) split into {n} children narrower than the time resolution {res:.3g}; "
                      "consider a smaller branching factor (rule R3)")
    return e


# ---------------------------------------------------------------------------
# torch helpers (used in training and evaluation)
# ---------------------------------------------------------------------------

def log_survival_torch(level_logps, path, plen, log_r):
    """log S(tau) from per-level log-probabilities read along tau's path.

    level_logps: list of L tensors [n, size_l] (rows of unused levels are ignored)
    path: [n, L] long (-1 padding), plen: [n] long, log_r: [n] float.
    """
    import torch
    n = path.shape[0]
    dev = path.device
    prefix = torch.zeros(n, device=dev, dtype=torch.float32)
    terms = []
    for lv, lp in enumerate(level_logps):
        lp = lp.float()
        active = plen > lv
        c = path[:, lv].clamp(min=0)
        idx = torch.arange(lp.shape[1], device=dev)
        later = idx[None, :] > c[:, None]
        logq_later = torch.logsumexp(lp.masked_fill(~later, NEG), dim=1)
        logq_later = torch.where(later.any(1), logq_later, torch.full_like(logq_later, NEG))
        terms.append(torch.where(active, prefix + logq_later, torch.full_like(prefix, NEG)))
        q = lp.gather(1, c[:, None]).squeeze(1)
        prefix = torch.where(active, prefix + q, prefix)
    terms.append(prefix + log_r.float())
    return torch.logsumexp(torch.stack(terms, 1), 1)


def log_survival_np(level_logps, path, plen, log_r):
    """Numpy version of log_survival_torch (evaluation)."""
    n = path.shape[0]
    prefix = np.zeros(n)
    terms = []
    for lv, lp in enumerate(level_logps):
        active = plen > lv
        c = np.clip(path[:, lv], 0, None)
        idx = np.arange(lp.shape[1])
        later = idx[None, :] > c[:, None]
        masked = np.where(later, lp, -np.inf)
        with np.errstate(divide="ignore"):
            mx = np.max(masked, axis=1, keepdims=True)
            mx = np.where(np.isfinite(mx), mx, 0.0)
            lql = np.log(np.sum(np.exp(masked - mx), axis=1)) + mx[:, 0]
        lql = np.where(later.any(1), lql, -np.inf)
        terms.append(np.where(active, prefix + lql, -np.inf))
        q = np.take_along_axis(lp, c[:, None], 1)[:, 0]
        prefix = np.where(active, prefix + q, prefix)
    terms.append(prefix + log_r)
    t = np.stack(terms, 1)
    mx = np.max(t, axis=1, keepdims=True)
    return np.log(np.sum(np.exp(t - mx), axis=1)) + mx[:, 0]
