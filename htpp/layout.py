"""From event sequences to LLM token sequences.

Each event is written as a block of tokens

    [c_1 ... c_n] [mark] [name words]? [<desc> review words </desc>]? [</e>]

The time tokens c_1..c_n are the event's path in the time code, the mark token is its
type. Which optional parts appear depends on the input condition (RQ3, Section 4):

    no_text    mark token only (type ID)
    names      mark token + the type's name ("Electronics")
    scrambled  mark token + a fixed meaningless string of similar length
    full       mark token + the event's own review
    shuffled   mark token + the review of another, earlier event of the same type
    text_only  the event's own review; the mark token is present (it is a training
               target) but hidden from all later positions by the attention mask

Targets are identical in every condition: the time tokens and the mark token of every
scored event, plus the final quiet gap. The model reads its probabilities at the
position *before* each target token. A window is a run of consecutive events; the
first `n_ctx` events are context only, so every event (except the very first event of a
sequence, whose waiting time is unknown) is scored exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import EventSeq
from .timecode import TimeCode

CONDITIONS = ("no_text", "names", "scrambled", "full", "shuffled", "text_only")


class EventVocab:
    """Ids of the new tokens. Base-model ids are [0, V); new ids start at V."""

    def __init__(self, tokenizer, tc: TimeCode, K: int, base_rows: int):
        self.tok = tokenizer
        self.tc = tc
        self.K = K
        self.V = int(base_rows)
        self.V_tok = len(tokenizer)
        self.new_names: list[str] = []

        def add(name):
            self.new_names.append(name)
            return self.V + len(self.new_names) - 1

        if tc.kind == "hier":
            self.level_ids = []
            for lv, b in enumerate(tc.branching):
                ids = [add(f"<t{lv + 1}_{j}>") for j in range(b)]
                if lv == 0:
                    ids.append(add("<t1_tail>"))
                self.level_ids.append(ids)
        else:
            digits = []
            for d in range(10):
                enc = tokenizer.encode(str(d), add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError(f"digit {d} is not a single token in this tokenizer")
                digits.append(enc[0])
            tail = add("<t_tail>")
            self.level_ids = [digits + ([tail] if lv == 0 else []) for lv in range(tc.L)]
        self.mark_ids = [add(f"<k{j}>") for j in range(K)]
        self.eoe = add("</e>")
        self.desc_open = add("<desc>")
        self.desc_close = add("</desc>")
        self.n_new = len(self.new_names)

    def name_of(self, i: int) -> str:
        return self.new_names[i - self.V] if i >= self.V else self.tok.decode([i])


@dataclass
class PreSeq:
    """A sequence with everything the layout needs, precomputed once."""
    sid: str
    k: np.ndarray
    tau: np.ndarray
    path: np.ndarray
    plen: np.ndarray
    log_u: np.ndarray
    log_r: np.ndarray
    desc: list | None
    shuf: list | None
    gap: float | None
    gap_path: np.ndarray | None
    gap_plen: int
    gap_log_r: float
    true_nll: float | None
    n_hist: int = 0


def preprocess(seqs: list[EventSeq], tc: TimeCode, tokenizer=None, desc_max_tokens: int = 48,
               need_text: bool = False) -> list[PreSeq]:
    out = []
    for s in seqs:
        tau = s.tau
        path, plen, fine = tc.encode(tau)
        log_u = tc.log_inbin(tau, fine)
        log_r = tc.log_after_share(tau, fine)
        desc = shuf = None
        if need_text and s.text is not None:
            desc = [tokenizer.encode(" " + (x or ""), add_special_tokens=False)[:desc_max_tokens] for x in s.text]
            if s.shuf is not None:
                shuf = [tokenizer.encode(" " + (x or ""), add_special_tokens=False)[:desc_max_tokens] for x in s.shuf]
        gap = None
        gp, gl, glr = None, 0, 0.0
        if s.t_end is not None and len(s.t) and s.t_end > s.t[-1]:
            gap = float(s.t_end - s.t[-1])
            p, l, f = tc.encode(np.array([gap]))
            gp, gl, glr = p[0], int(l[0]), float(tc.log_after_share(np.array([gap]), f)[0])
        out.append(PreSeq(s.sid, s.k, tau, path, plen, log_u, log_r, desc, shuf, gap, gp, gl, glr, s.true_nll,
                          int(getattr(s, "n_hist", 0) or 0)))
    return out


@dataclass
class LayoutOptions:
    condition: str = "no_text"
    gen: bool = False                 # score the description tokens (FULL+GEN)
    prompt_ids: tuple = ()
    name_ids: tuple = ()              # tokenized type names
    scrambled_ids: tuple = ()
    budget: str = "event"             # "event" or "token"
    max_events: int = 64
    max_tokens: int = 2048
    ctx_frac: float = 0.5
    predict_depth: int = 0            # >0: append branch tokens for time prediction


def event_parts(o: LayoutOptions, ps: PreSeq, j: int):
    """(extra tokens after the mark, description tokens or None, hide mark)."""
    c = o.condition
    k = int(ps.k[j])
    if c == "no_text":
        return (), None, False
    if c == "names":
        return o.name_ids[k], None, False
    if c == "scrambled":
        return o.scrambled_ids[k], None, False
    if c == "full":
        return (), (ps.desc[j] if ps.desc is not None else []), False
    if c == "shuffled":
        return (), (ps.shuf[j] if ps.shuf is not None else []), False
    if c == "text_only":
        return (), (ps.desc[j] if ps.desc is not None else []), True
    raise ValueError(f"unknown condition {c!r}")


def block_len(o: LayoutOptions, ps: PreSeq, j: int) -> int:
    extra, desc, _ = event_parts(o, ps, j)
    n = int(ps.plen[j]) + 1 + len(extra) + 1
    if desc is not None:
        n += len(desc) + 2
    return n


@dataclass
class Window:
    seq: int
    start: int
    n_ctx: int
    end: int
    with_gap: bool
    n_tokens: int


def make_windows(pre: list[PreSeq], o: LayoutOptions, with_gap: bool = True) -> list[Window]:
    """Cover every sequence with windows so that events max(1, n_hist)..n-1 are scored
    exactly once (earlier events are history only).
    Event budget: at most max_events events per window, the first ctx_frac of them
    context. Token budget: at most max_tokens tokens, with up to ctx_frac of them used by
    the most recent earlier events as context."""
    wins = []
    P = len(o.prompt_ids)
    for si, ps in enumerate(pre):
        n = len(ps.k)
        first = max(1, int(ps.n_hist))
        if n < 2 or first >= n:
            continue
        lens = [block_len(o, ps, j) for j in range(n)]
        gap_tokens = max(ps.gap_plen - 1, 0) if (with_gap and ps.gap is not None) else 0
        if o.budget == "event":
            E = max(2, int(o.max_events))
            C = min(E - 1, max(1, int(round(E * o.ctx_frac))))
            start = max(0, first - C)
            n_ctx = first - start
            while True:
                end = min(start + E, n)
                last = end == n
                g = with_gap and ps.gap is not None and last
                wins.append(Window(si, start, n_ctx, end, g, P + sum(lens[start:end]) + (gap_tokens if g else 0)))
                if last:
                    break
                start, n_ctx = end - C, C
        else:
            B = int(o.max_tokens)
            Bc = int(B * o.ctx_frac)
            nxt = first                  # first event to score
            while nxt < n:
                # context: most recent earlier events fitting in Bc tokens (at least one)
                start = nxt - 1
                used = lens[start]
                while start - 1 >= 0 and used + lens[start - 1] <= Bc:
                    start -= 1
                    used += lens[start]
                end = nxt
                while end < n and (used + lens[end] <= B or end == nxt):
                    used += lens[end]
                    end += 1
                last = end == n
                g = with_gap and ps.gap is not None and last
                wins.append(Window(si, start, nxt - start, end, g, P + used + (gap_tokens if g else 0)))
                nxt = end
    return wins


def build_layout(ps: PreSeq, w: Window, V: EventVocab, o: LayoutOptions, branch_events: set | None = None) -> dict:
    """Token ids and read positions for one window (see module docstring)."""
    tc = V.tc
    L = tc.L
    ids = list(o.prompt_ids)
    hide = []
    t_reads = [[] for _ in range(L)]   # (pos, target index, event)
    m_reads = []                       # (pos, target type, event)
    d_reads = []                       # (pos, target token id or -1 for </desc>, event)
    events = []
    branch_at = []                     # (history-end position, event)
    for j in range(w.start, w.end):
        scored = j >= w.start + w.n_ctx
        ev = len(events)
        s = len(ids)
        pl = int(ps.plen[j])
        for lv in range(pl):
            c = int(ps.path[j, lv])
            if scored:
                t_reads[lv].append((s - 1 + lv, c, ev))
            ids.append(V.level_ids[lv][c])
        if scored:
            m_reads.append((s + pl - 1, int(ps.k[j]), ev))
        mark_pos = len(ids)
        ids.append(V.mark_ids[int(ps.k[j])])
        extra, desc, hide_mark = event_parts(o, ps, j)
        if hide_mark:
            hide.append(mark_pos)
        ids.extend(extra)
        if desc is not None:
            p_d = len(ids)
            ids.append(V.desc_open)
            ids.extend(desc)
            ids.append(V.desc_close)
            if scored and o.gen:
                for i, tok in enumerate(list(desc) + [-1]):
                    d_reads.append((p_d + i, int(tok), ev))
        ids.append(V.eoe)
        if scored:
            events.append({"seq": ps.sid, "idx": j, "tau": float(ps.tau[j]), "k": int(ps.k[j]),
                           "prev_k": int(ps.k[j - 1]), "hist_len": j, "ctx_len": j - w.start,
                           "plen": pl, "path": ps.path[j].tolist(), "log_u": float(ps.log_u[j]),
                           "log_r": float(ps.log_r[j]), "n_desc": (len(desc) + 1) if (desc is not None and o.gen) else 0})
            if branch_events is not None and (ps.sid, j) in branch_events:
                branch_at.append((s - 1, ev))
    gap = None
    if w.with_gap and ps.gap is not None:
        p0 = len(ids) - 1
        reads = [p0 + lv for lv in range(ps.gap_plen)]
        for lv in range(ps.gap_plen - 1):
            ids.append(V.level_ids[lv][int(ps.gap_path[lv])])
        gap = {"reads": reads, "path": ps.gap_path.tolist(), "plen": ps.gap_plen, "log_r": ps.gap_log_r,
               "gap": ps.gap, "seq": ps.sid, "true_nll": ps.true_nll}
    n_main = len(ids)
    positions = list(range(n_main))
    branches = []      # (event, {level: [read positions]}) ; tokens appended below
    branch_tok = []    # (position, history-end position, parent branch position or -1)
    depth = min(o.predict_depth, L) if o.predict_depth else 0
    for (p, ev) in branch_at:
        rows = {0: [p]}
        parents = [(p, -1)]
        for lv in range(depth - 1):
            new = []
            for (pp, par_is_branch) in parents:
                for c in range(tc.branching[lv]):
                    pos = len(ids)
                    ids.append(V.level_ids[lv][c])
                    positions.append(positions[p] + lv + 1)
                    branch_tok.append((pos, p, pp if par_is_branch >= 0 else -1))
                    new.append((pos, 1))
            parents = new
            rows[lv + 1] = [pp for pp, _ in parents]
        branches.append((ev, rows))
    return {"ids": ids, "positions": positions, "n_main": n_main, "hide": hide, "t_reads": t_reads,
            "m_reads": m_reads, "d_reads": d_reads, "events": events, "gap": gap,
            "branches": branches, "branch_tok": branch_tok, "seq_true_nll": ps.true_nll}


def _custom_mask(lay: dict, T: int) -> np.ndarray:
    """Boolean [T, T] attention mask (True = may attend) for windows that hide mark
    tokens or carry branch tokens."""
    n = len(lay["ids"])
    nm = lay["n_main"]
    M = np.zeros((T, T), dtype=bool)
    M[:nm, :nm] = np.tril(np.ones((nm, nm), dtype=bool))
    for h in lay["hide"]:
        M[h + 1:nm, h] = False
    anc = {}
    for (pos, p, parent) in lay["branch_tok"]:
        row = M[p].copy()
        row[p] = True
        chain = anc.get(parent, []) + ([parent] if parent >= 0 else [])
        for a in chain:
            row[a] = True
        row[pos] = True
        M[pos] = row
        anc[pos] = chain
    for i in range(n, T):
        M[i, i] = True
    return M


def collate(layouts: list[dict], pad_id: int, n_levels: int, dtype=torch.float32) -> dict:
    B = len(layouts)
    T = max(len(x["ids"]) for x in layouts)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    pos = torch.zeros((B, T), dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    need4d = any(x["hide"] or x["branch_tok"] for x in layouts)
    mask4d = None
    if need4d:
        mask4d = torch.zeros((B, 1, T, T), dtype=dtype)
        neg = torch.finfo(dtype).min
    tr = [[] for _ in range(n_levels)]
    mr, dr = [], []
    events, gaps, branches = [], [], []
    g_reads = [[] for _ in range(n_levels)]
    for b, x in enumerate(layouts):
        n = len(x["ids"])
        ids[b, :n] = torch.tensor(x["ids"])
        pos[b, :n] = torch.tensor(x["positions"])
        att[b, :n] = 1
        if need4d:
            M = torch.from_numpy(_custom_mask(x, T))
            mask4d[b, 0].masked_fill_(~M, neg)
        off = len(events)
        for lv in range(n_levels):
            tr[lv].extend((b, p, c, off + e) for (p, c, e) in x["t_reads"][lv])
        mr.extend((b, p, c, off + e) for (p, c, e) in x["m_reads"])
        dr.extend((b, p, c, off + e) for (p, c, e) in x["d_reads"])
        events.extend(x["events"])
        if x["gap"] is not None:
            gi = len(gaps)
            for lv, p in enumerate(x["gap"]["reads"]):
                g_reads[lv].append((b, p, gi))
            gaps.append(x["gap"])
        for (ev, rows) in x["branches"]:
            branches.append((off + ev, b, rows))

    def t4(rows):
        if not rows:
            return None
        a = torch.tensor(rows, dtype=torch.long)
        return {"b": a[:, 0], "pos": a[:, 1], "tgt": a[:, 2], "ev": a[:, 3]}

    out = {"input_ids": ids, "position_ids": pos, "attention_mask": att, "mask4d": mask4d,
           "time_reads": [t4(r) for r in tr], "mark_reads": t4(mr), "desc_reads": t4(dr),
           "events": events, "gaps": gaps, "branches": branches, "n_events": len(events)}
    out["ev_log_u"] = torch.tensor([e["log_u"] for e in events], dtype=torch.float32)
    if gaps:
        L = n_levels
        gp = torch.full((len(gaps), L), -1, dtype=torch.long)
        for i, g in enumerate(gaps):
            gp[i, : len(g["path"])] = torch.tensor(g["path"])
        out["gap_path"] = gp
        out["gap_plen"] = torch.tensor([g["plen"] for g in gaps], dtype=torch.long)
        out["gap_log_r"] = torch.tensor([g["log_r"] for g in gaps], dtype=torch.float32)
        out["gap_reads"] = []
        for lv in range(L):
            if g_reads[lv]:
                a = torch.tensor(g_reads[lv], dtype=torch.long)
                out["gap_reads"].append({"b": a[:, 0], "pos": a[:, 1], "row": a[:, 2]})
            else:
                out["gap_reads"].append(None)
    return out


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, pre: list[PreSeq], wins: list[Window], vocab: EventVocab, opts: LayoutOptions,
                 branch_events: set | None = None):
        self.pre, self.wins, self.vocab, self.opts = pre, wins, vocab, opts
        self.branch_events = branch_events

    def __len__(self):
        return len(self.wins)

    def __getitem__(self, i):
        w = self.wins[i]
        return build_layout(self.pre[w.seq], w, self.vocab, self.opts, self.branch_events)


def token_batches(wins: list[Window], max_tokens: int, max_bs: int, shuffle: bool, rng: np.random.Generator,
                  extra_per_window: list[int] | None = None) -> list[list[int]]:
    """Group windows of similar length so that padded batch size x length <= max_tokens."""
    lens = np.array([w.n_tokens + (extra_per_window[i] if extra_per_window else 0) for i, w in enumerate(wins)])
    order = np.argsort(lens, kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = int(lens[i])
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) >= max_bs):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = L
        cur.append(int(i))
        cur_max = new_max
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches
