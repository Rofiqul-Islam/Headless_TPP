"""Event-sequence data: storage format, loaders, censoring, and synthetic sequences.

On disk a prepared dataset is a folder with
    meta.json                 K, type names, scrambled names, parents, time unit, resolution
    train.jsonl / val.jsonl / test.jsonl
one sequence per line:
    {"id": str, "t": [abs. times], "k": [types], "text": [str] | null, "shuf": [str] | null,
     "t_end": float | null, "true_nll": float | null}
`t_end` is the end of the observation window (the final quiet gap is t_end - t[-1]);
`shuf` holds, for each event, the review of another, earlier event of the same type
(SHUFFLED input); `true_nll` is the true negative log-likelihood of the scored part of
the sequence when it is known (synthetic data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import load_json, read_jsonl, save_json, write_jsonl


@dataclass
class EventSeq:
    sid: str
    t: np.ndarray
    k: np.ndarray
    text: list | None = None
    shuf: list | None = None
    t_end: float | None = None
    true_nll: float | None = None
    n_hist: int = 0            # leading events that are history only (chronological splits)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=np.float64)
        self.k = np.asarray(self.k, dtype=np.int64)

    def __len__(self):
        return len(self.t)

    @property
    def tau(self) -> np.ndarray:
        """Waiting times; the first event's waiting time is unknown and set to 0."""
        return np.diff(self.t, prepend=self.t[0]) if len(self.t) else self.t

    def to_row(self) -> dict:
        r = {"id": self.sid, "t": self.t.tolist(), "k": self.k.tolist(), "text": self.text,
             "shuf": self.shuf, "t_end": self.t_end, "true_nll": self.true_nll, "n_hist": int(self.n_hist)}
        r.update(self.extra)
        return r

    @classmethod
    def from_row(cls, r: dict) -> "EventSeq":
        extra = {k: v for k, v in r.items() if k not in {"id", "t", "k", "text", "shuf", "t_end", "true_nll", "n_hist"}}
        return cls(sid=str(r["id"]), t=r["t"], k=r["k"], text=r.get("text"), shuf=r.get("shuf"),
                   t_end=r.get("t_end"), true_nll=r.get("true_nll"), n_hist=int(r.get("n_hist") or 0), extra=extra)


def save_dataset(out_dir: str | Path, splits: dict[str, list[EventSeq]], meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, seqs in splits.items():
        write_jsonl([s.to_row() for s in seqs], out_dir / f"{name}.jsonl")
    stats = {name: {"sequences": len(seqs), "events": int(sum(len(s) for s in seqs))} for name, seqs in splits.items()}
    meta = dict(meta)
    meta["stats"] = stats
    save_json(meta, out_dir / "meta.json")


def load_split(data_dir: str | Path, split: str, max_seqs: int | None = None, frac: float | None = None,
               seed: int = 0) -> list[EventSeq]:
    rows = read_jsonl(Path(data_dir) / f"{split}.jsonl")
    if frac is not None and 0 < float(frac) < 1:   # e.g. keep 10% / 30% of the training users
        rng = np.random.default_rng(seed)
        keep = rng.permutation(len(rows))[: max(1, int(round(float(frac) * len(rows))))]
        rows = [rows[i] for i in sorted(keep)]
    if max_seqs:
        rows = rows[: int(max_seqs)]
    return [EventSeq.from_row(r) for r in rows]


def has_split(data_dir: str | Path, split: str) -> bool:
    return (Path(data_dir) / f"{split}.jsonl").exists()


def load_meta(data_dir: str | Path) -> dict:
    return load_json(Path(data_dir) / "meta.json")


# ----------------------------------------------------------------------------
# censoring: create an observation end (and hence a final quiet gap)
# ----------------------------------------------------------------------------

def censor_random_cut(seqs: list[EventSeq], rng: np.random.Generator, min_events: int = 3) -> list[EventSeq]:
    """Cut each sequence at a random time inside its span; events after the cut are
    dropped and the cut becomes the observation end t_end."""
    out = []
    for s in seqs:
        if len(s) < min_events + 1:
            continue
        lo = s.t[min_events - 1]
        hi = s.t[-1]
        if hi <= lo:
            continue
        cut = rng.uniform(lo, hi)
        keep = s.t <= cut
        ns = EventSeq(s.sid, s.t[keep], s.k[keep],
                      None if s.text is None else [x for x, m in zip(s.text, keep) if m],
                      None if s.shuf is None else [x for x, m in zip(s.shuf, keep) if m],
                      t_end=float(cut), extra=dict(s.extra))
        if len(ns) >= min_events:
            out.append(ns)
    return out


def split_long(seqs: list[EventSeq], max_len: int) -> list[EventSeq]:
    """Cut long sequences into consecutive pieces of at most max_len events."""
    out = []
    for s in seqs:
        if len(s) <= max_len:
            out.append(s)
            continue
        for p, a in enumerate(range(0, len(s), max_len)):
            b = min(a + max_len, len(s))
            if b - a < 3:
                continue
            out.append(EventSeq(f"{s.sid}#{p}", s.t[a:b], s.k[a:b],
                                None if s.text is None else s.text[a:b],
                                None if s.shuf is None else s.shuf[a:b],
                                t_end=None, extra=dict(s.extra)))
    return out


# ----------------------------------------------------------------------------
# synthetic sequences for stage 1 (event grammar)
# ----------------------------------------------------------------------------

def synthetic_grammar_sequences(n: int, K: int, type_probs: np.ndarray, mean_tau: float,
                                rng: np.random.Generator, length: int = 32) -> list[EventSeq]:
    """Poisson and self-exciting sequences with the dataset's type frequencies and mean
    waiting time, used to teach the LLM the token format (stage 1)."""
    seqs = []
    p = np.asarray(type_probs, float)
    p = p / p.sum()
    for i in range(n):
        if i % 2 == 0:   # Poisson
            t = np.cumsum(rng.exponential(mean_tau, length))
        else:            # 1-d Hawkes ground process (exponential kernel), branching ratio 0.5
            base, decay = 0.5 / mean_tau, 2.0 / mean_tau
            now, excite, times = 0.0, 0.0, []
            while len(times) < length:   # Ogata thinning; the intensity only decays between events
                lam_bar = base + excite
                dt = rng.exponential(1.0 / lam_bar)
                now += dt
                excite *= np.exp(-decay * dt)
                if rng.random() * lam_bar <= base + excite:
                    times.append(now)
                    excite += 0.5 * decay
            t = np.asarray(times)
        k = rng.choice(K, size=length, p=p)
        seqs.append(EventSeq(f"syn{i}", t, k))
    return seqs
