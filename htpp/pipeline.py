"""Builds everything one run needs from a config: data, time code, vocabulary,
token layouts and windows."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import EventSeq, has_split, load_meta, load_split, synthetic_grammar_sequences
from .layout import (CONDITIONS, EventVocab, LayoutOptions, PreSeq, Window, make_windows, preprocess)
from .timecode import TimeCode
from .utils import stable_hash

TEXT_CONDITIONS = {"full", "shuffled", "text_only"}


@dataclass
class Setup:
    cfg: dict
    meta: dict
    tokenizer: object
    tc: TimeCode
    vocab: EventVocab
    opts: LayoutOptions
    pre: dict = field(default_factory=dict)      # split -> [PreSeq]
    wins: dict = field(default_factory=dict)     # split -> [Window]
    raw: dict = field(default_factory=dict)      # split -> [EventSeq]
    name_ids: list = field(default_factory=list)


def base_vocab_rows(model_name: str) -> int:
    from transformers import AutoConfig
    conf = AutoConfig.from_pretrained(model_name)
    v = getattr(conf, "vocab_size", None)
    if v is None and hasattr(conf, "text_config"):
        v = conf.text_config.vocab_size
    return int(v)


def build_timecode(cfg: dict, train: list[EventSeq], meta: dict) -> TimeCode:
    tcfg = cfg["timecode"]
    taus = np.concatenate([s.tau[1:] for s in train if len(s) > 1])
    if tcfg.get("kind", "hier") == "digits":
        dc = tcfg.get("digits", {})
        n_int, n_dec = int(dc.get("n_int", 3)), int(dc.get("n_dec", 2))
        scale = dc.get("scale")
        if scale is None:   # choose a power of ten so that 99% of waits fit in n_int digits
            q = float(np.quantile(taus, 0.99))
            scale = 10.0 ** np.ceil(np.log10(max(q, 1e-12) / (10 ** n_int)))
        return TimeCode.digits(n_int, n_dec, float(scale), taus)
    res = tcfg.get("time_resolution")
    if res is None:
        res = meta.get("time_resolution")
    return TimeCode.fit(taus, branching=tuple(tcfg.get("branching", (8, 8, 8))), p_tail=float(tcfg.get("p_tail", 0.01)),
                        resolution=res, split=tcfg.get("split", "quantile"))


def setup(cfg: dict, log=None, splits=("train", "val", "test", "test_unseen")) -> Setup:
    from transformers import AutoTokenizer
    dcfg = cfg["data"]
    meta = load_meta(dcfg["dir"])
    cond = dcfg.get("condition", "no_text")
    if cond not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    gen = bool(cfg["train"].get("gen", False))
    raw = {}
    splits = [s for s in splits if has_split(dcfg["dir"], s)]
    for s in splits:
        mx = dcfg.get(f"max_{s}_seqs")
        frac = dcfg.get("train_frac") if s == "train" else None
        raw[s] = load_split(dcfg["dir"], s, mx, frac, int(cfg["train"].get("seed", 1)))
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tc = build_timecode(cfg, raw["train"], meta)
    K = int(meta["K"])
    vocab = EventVocab(tok, tc, K, base_vocab_rows(cfg["model"]["name"]))
    names = meta.get("type_names") or [f"type {j}" for j in range(K)]
    scr = meta.get("scrambled_names") or [f"zz{j}" for j in range(K)]
    name_ids = [tuple(tok.encode(" " + n, add_special_tokens=False)) for n in names]
    scr_ids = [tuple(tok.encode(" " + n, add_special_tokens=False)) for n in scr]
    opts = LayoutOptions(condition=cond, gen=gen, prompt_ids=tuple(tok.encode(dcfg.get("prompt", ""), add_special_tokens=False)),
                         name_ids=tuple(name_ids), scrambled_ids=tuple(scr_ids), budget=dcfg.get("budget", "event"),
                         max_events=int(dcfg.get("max_events", 64)), max_tokens=int(dcfg.get("max_tokens", 2048)),
                         ctx_frac=float(dcfg.get("ctx_frac", 0.5)))
    need_text = cond in TEXT_CONDITIONS or gen
    if need_text and not meta.get("has_text", False):
        raise ValueError(f"condition {cond!r} needs a dataset with text")
    S = Setup(cfg, meta, tok, tc, vocab, opts, raw=raw, name_ids=[list(x) for x in name_ids])
    for s in splits:
        S.pre[s] = preprocess(raw[s], tc, tok, int(dcfg.get("desc_max_tokens", 48)), need_text)
        S.wins[s] = make_windows(S.pre[s], opts, with_gap=bool(dcfg.get("silence", True)))
    if log:
        n_ev = {s: int(sum(len(p.k) - max(1, p.n_hist) for p in S.pre[s] if len(p.k) > max(1, p.n_hist))) for s in splits}
        log.info(f"data {dcfg['dir']} | condition={cond} budget={opts.budget} | K={K} | scored events {n_ev}")
        log.info(f"time code: kind={tc.kind} branching={tc.branching} a_max={tc.a_max:.4g} beta={tc.beta:.4g} "
                 f"resolution={tc.resolution:.3g} | new tokens={vocab.n_new}")
        lens = [w.n_tokens for w in S.wins["train"]]
        log.info(f"train windows={len(S.wins['train'])} tokens/window mean={np.mean(lens):.0f} max={np.max(lens)}")
    return S


def grammar_windows(S: Setup, n: int, seed: int) -> tuple[list[PreSeq], list[Window], LayoutOptions]:
    """Synthetic Poisson/Hawkes sequences for stage 1, in the no-text format."""
    rng = np.random.default_rng(seed)
    K = S.vocab.K
    cnt = np.bincount(np.concatenate([s.k for s in S.raw["train"]]), minlength=K).astype(float) + 1.0
    taus = np.concatenate([s.tau[1:] for s in S.raw["train"] if len(s) > 1])
    seqs = synthetic_grammar_sequences(n, K, cnt / cnt.sum(), float(np.mean(taus)), rng,
                                       length=min(32, max(4, int(S.opts.max_events))))
    pre = preprocess(seqs, S.tc)
    o = LayoutOptions(condition="no_text", prompt_ids=S.opts.prompt_ids, budget="event",
                      max_events=S.opts.max_events, ctx_frac=S.opts.ctx_frac)
    return pre, make_windows(pre, o, with_gap=False), o


def pick_events(pre: list[PreSeq], n: int, seed: int) -> set:
    """A deterministic subset of scored events (sequence id, index)."""
    cand = [(p.sid, j) for p in pre for j in range(max(1, p.n_hist), len(p.k))]
    if n is None or n <= 0 or n >= len(cand):
        return set(cand)
    cand.sort(key=lambda x: stable_hash(seed, x[0], x[1]))
    return set(cand[:n])
