"""Writing the next event's description (Section 4.3, FULL+GEN): sampling with
nucleus sampling given the history and the true time and type, ROUGE-L against the
real review, and a memorisation check (long n-gram overlap with the training text)."""
from __future__ import annotations

import numpy as np
import torch

from .layout import Window, build_layout
from .pipeline import Setup, pick_events
from .utils import stable_hash


def rouge_l(pred: str, ref: str) -> float:
    a, b = pred.lower().split(), ref.lower().split()
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def _ngram_hashes(text: str, n: int) -> list[int]:
    w = text.lower().split()
    return [stable_hash(*w[i:i + n]) for i in range(len(w) - n + 1)]


def ngram_index(texts, n=8, max_texts=500_000) -> np.ndarray:
    hs = []
    for t in texts[:max_texts]:
        hs.extend(_ngram_hashes(t, n))
    return np.unique(np.array(hs, dtype=np.int64))


@torch.no_grad()
def sample_description(model, prefix_ids: list[int], max_new: int, top_p: float, rng: torch.Generator) -> list[int]:
    dev = model.device_
    ids = torch.tensor([prefix_ids], device=dev)
    x = model.embed_ids(ids)
    pos = torch.arange(len(prefix_ids), device=dev)[None]
    out = model.inner(inputs_embeds=x, position_ids=pos, use_cache=True)
    past = out.past_key_values
    h = out.last_hidden_state[:, -1]
    Vt = model.vocab.V_tok
    W = model.out_w[:Vt]
    close = model.new_out[model.vocab.desc_close - model.V]
    toks = []
    for step in range(max_new + 1):
        logits = torch.cat([(h.to(W.dtype) @ W.T).float(), (h.float() @ close)[:, None]], 1)[0]
        probs = torch.softmax(logits, -1)
        sp, si = torch.sort(probs, descending=True)
        keep = torch.cumsum(sp, 0) - sp < top_p
        sp = sp * keep
        choice = si[torch.multinomial(sp / sp.sum(), 1, generator=rng)].item()
        if choice == Vt or step == max_new:
            break
        toks.append(choice)
        x = model.embed_ids(torch.tensor([[choice]], device=dev))
        p = torch.tensor([[len(prefix_ids) + len(toks) - 1]], device=dev)
        out = model.inner(inputs_embeds=x, position_ids=p, past_key_values=past, use_cache=True)
        past = out.past_key_values
        h = out.last_hidden_state[:, -1]
    return toks


@torch.no_grad()
def generate_eval(model, S: Setup, n_events: int, top_p: float = 0.9, seed: int = 0, max_new: int | None = None) -> dict:
    """Generate the description of `n_events` test events and score them."""
    model.eval()
    pre = S.pre["test"]
    chosen = pick_events(pre, n_events, seed)
    by_sid = {p.sid: (i, p) for i, p in enumerate(pre)}
    max_new = max_new or int(S.cfg["data"].get("desc_max_tokens", 48))
    g = torch.Generator(device=model.device_).manual_seed(seed)
    idx = ngram_index([x for s in S.raw["train"] for x in (s.text or []) if x])
    rows = []
    for (sid, j) in sorted(chosen, key=lambda x: stable_hash(seed, *x)):
        si, ps = by_sid[sid]
        E = int(S.opts.max_events)
        start = max(0, j - E + 1)
        w = Window(si, start, j - start, j + 1, False, 0)
        lay = build_layout(ps, w, S.vocab, S.opts)
        ids = lay["ids"]
        cut = len(ids) - 1 - ids[::-1].index(S.vocab.desc_open)
        prefix = ids[: cut + 1]
        toks = sample_description(model, prefix, max_new, top_p, g)
        text = S.tokenizer.decode(toks, skip_special_tokens=True).strip()
        ref = S.tokenizer.decode(ps.desc[j], skip_special_tokens=True).strip() if ps.desc else ""
        hs = np.array(_ngram_hashes(text, 8), dtype=np.int64)
        overlap = float(np.isin(hs, idx).mean()) if len(hs) else 0.0
        rows.append({"seq": sid, "idx": j, "generated": text, "reference": ref, "rouge_l": rouge_l(text, ref),
                     "ngram8_overlap": overlap})
    model.train()
    res = {"n": len(rows), "rouge_l": float(np.mean([r["rouge_l"] for r in rows])) if rows else float("nan"),
           "ngram8_overlap": float(np.mean([r["ngram8_overlap"] for r in rows])) if rows else float("nan")}
    return {"summary": res, "rows": rows}
