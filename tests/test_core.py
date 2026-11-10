"""Core correctness tests. Run:  python -m pytest -q tests/test_core.py
(the model tests need tests/make_fixtures.py to have been run)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from htpp.timecode import TimeCode, log_survival_np  # noqa: E402

HERE = Path(__file__).resolve().parent


def _random_tree(tc: TimeCode, rng):
    """Random level distributions for every prefix (as the expanded tree would give)."""
    lv = []
    n_pref = 1
    for l, b in enumerate(tc.branching):
        size = tc.level_size(l)
        lv.append(np.log(rng.dirichlet(np.ones(size), size=n_pref)))
        n_pref *= b
    return lv


def _path_levels(tc, tree, path, plen):
    """Per-level log-prob rows along a path (what the model reads at the true path)."""
    rows = []
    idx = 0
    for l in range(tc.L):
        if l < plen:
            rows.append(tree[l][idx])
            if l + 1 < tc.L:
                idx = idx * tc.branching[l] + path[l]
        else:
            rows.append(np.zeros(tc.level_size(l)))
    return rows


@pytest.mark.parametrize("kind", ["hier", "digits"])
def test_density_integrates_and_survival_matches(kind):
    rng = np.random.default_rng(0)
    taus = rng.exponential(5.0, 5000)
    tc = TimeCode.fit(taus, (4, 3, 2), p_tail=0.05, resolution=1e-3) if kind == "hier" else \
        TimeCode.digits(1, 2, 1.0, taus)
    tree = _random_tree(tc, rng)
    p_fine, p_tail = tc.fine_probs(tree)
    assert abs(p_fine.sum() + p_tail - 1.0) < 1e-9              # valid next-event distribution (R1, R2)
    # density integrates to one: finite bins exactly, tail analytically
    assert abs(np.sum(p_fine / tc.widths * tc.widths) + p_tail - 1.0) < 1e-9
    # survival from the path formula equals 1 - CDF from the fine-bin probabilities
    test_taus = np.concatenate([rng.uniform(0, tc.a_max, 50), tc.a_max + rng.exponential(2.0, 10), [0.0]])
    paths, plens, fine = tc.encode(test_taus)
    log_r = tc.log_after_share(test_taus, fine)
    for i, t in enumerate(test_taus):
        rows = _path_levels(tc, tree, paths[i], plens[i])
        lS = log_survival_np([r[None] for r in rows], paths[i][None], plens[i][None], log_r[i:i + 1])[0]
        if fine[i] >= 0:
            m = fine[i]
            cdf = p_fine[:m].sum() + p_fine[m] * (t - tc.edges[m]) / tc.widths[m]
        else:
            cdf = p_fine.sum() + p_tail * (1 - np.exp(-tc.beta * (t - tc.a_max)))
        assert abs(np.exp(lS) - (1 - cdf)) < 1e-8
        # density from the path equals fine-bin probability / width
        lp = sum(rows[l][paths[i][l]] for l in range(plens[i])) + tc.log_inbin(np.array([t]), fine[i:i + 1])[0]
        dens = p_fine[m] / tc.widths[m] if fine[i] >= 0 else p_tail * tc.beta * np.exp(-tc.beta * (t - tc.a_max))
        assert abs(np.exp(lp) - dens) < 1e-8 * max(1.0, dens)


def test_bins_respect_resolution():
    taus = np.concatenate([np.zeros(500), np.full(500, 1.0), np.random.default_rng(1).exponential(3.0, 1000)])
    tc = TimeCode.fit(taus, (8, 8), p_tail=0.01, resolution=0.01)
    assert tc.widths.min() >= 0.01 * (1 - 1e-9)


def test_mean_median():
    tc = TimeCode.fit(np.random.default_rng(2).exponential(1.0, 20000), (8, 8, 8), p_tail=0.01, resolution=1e-4)
    # a distribution that is uniform over fine bins in proportion to an exponential
    mids = tc.bin_mids()
    p = np.exp(-mids) * tc.widths
    p_tail = np.exp(-tc.a_max)
    s = p.sum() + p_tail
    mean, med = tc.mean_median(p / s, p_tail / s)
    assert abs(mean - 1.0) < 0.05 and abs(med - np.log(2)) < 0.05


def test_windows_score_each_event_once():
    from htpp.data import EventSeq
    from htpp.layout import LayoutOptions, make_windows, preprocess
    rng = np.random.default_rng(3)
    seqs = [EventSeq(f"s{i}", np.cumsum(rng.exponential(1, n)), rng.integers(0, 3, n), t_end=None, n_hist=h)
            for i, (n, h) in enumerate([(2, 0), (5, 0), (17, 0), (40, 0), (101, 0), (60, 45), (30, 3)])]
    tc = TimeCode.fit(np.concatenate([s.tau[1:] for s in seqs]), (4, 4), resolution=1e-4)
    pre = preprocess(seqs, tc)
    for budget, kw in (("event", {"max_events": 8}), ("token", {"max_tokens": 37})):
        o = LayoutOptions(budget=budget, **kw)
        wins = make_windows(pre, o)
        for si, ps in enumerate(pre):
            scored = []
            for w in wins:
                if w.seq == si:
                    scored.extend(range(w.start + w.n_ctx, w.end))
            assert sorted(scored) == list(range(max(1, ps.n_hist), len(ps.k))), (budget, si)


# ------------------------------------------------------------------ model tests
tiny = HERE / "tiny_model"
needs_tiny = pytest.mark.skipif(not tiny.exists(), reason="run tests/make_fixtures.py first")


def _setup(cond="full", extra=()):
    from htpp.pipeline import setup
    from htpp.utils import load_config
    cfg = load_config(str(ROOT / "configs" / "tiny.yaml"),
                      [f"model.name={tiny}", f"data.dir={HERE / 'data' / 'text_tiny'}", f"data.condition={cond}", *extra])
    return cfg, setup(cfg)


@needs_tiny
def test_branch_expansion_matches_naive_forward():
    import torch
    from htpp.layout import build_layout, collate
    from htpp.model import HeadlessTPP
    cfg, S = _setup("no_text")
    model = HeadlessTPP(cfg["model"], S.vocab, None, "cpu").eval()
    ps = S.pre["test"][0]
    w = S.wins["test"][0]
    j = w.start + w.n_ctx
    o = replace(S.opts, predict_depth=3)
    lay = build_layout(ps, w, S.vocab, o, {(ps.sid, j)})
    b = collate([lay], S.tokenizer.pad_token_id, S.tc.L)
    with torch.no_grad():
        out = model.compute(b, keep=True)
    ev, per = out["branch_logits"][0]
    # naive: history up to the end of event j-1, then an explicit prefix (c1, c2)
    hist_end = lay["branches"][0][1][0][0]
    ids = lay["ids"][: hist_end + 1]
    for c1 in range(S.tc.branching[0]):
        for c2 in range(S.tc.branching[1]):
            seq = ids + [S.vocab.level_ids[0][c1], S.vocab.level_ids[1][c2]]
            x = torch.tensor([seq])
            bb = {"input_ids": x, "position_ids": torch.arange(len(seq))[None], "attention_mask": torch.ones_like(x),
                  "mask4d": None}
            with torch.no_grad():
                h = model.hidden(bb)
                l2 = model.level_logits(h[0, -2:-1], 1)[0]
                l3 = model.level_logits(h[0, -1:], 2)[0]
            assert torch.allclose(per[1][c1], l2, atol=1e-4)
            assert torch.allclose(per[2][c1 * S.tc.branching[1] + c2], l3, atol=1e-4)


@needs_tiny
def test_text_only_hides_type_tokens():
    import torch
    from htpp.layout import build_layout, collate
    from htpp.model import HeadlessTPP
    cfg, S = _setup("text_only")
    model = HeadlessTPP(cfg["model"], S.vocab, None, "cpu").eval()
    ps = S.pre["test"][1]
    w = S.wins["test"][[x.seq for x in S.wins["test"]].index(1)]
    lay = build_layout(ps, w, S.vocab, S.opts)
    b = collate([lay], S.tokenizer.pad_token_id, S.tc.L)
    ps2 = replace(ps, k=(ps.k + 1) % S.vocab.K)          # change every type in the history
    lay2 = build_layout(ps2, w, S.vocab, S.opts)
    b2 = collate([lay2], S.tokenizer.pad_token_id, S.tc.L)
    with torch.no_grad():
        o1 = model.compute(b, keep=True)
        o2 = model.compute(b2, keep=True)
    # time-token distributions never see the history's types ...
    for l in range(S.tc.L):
        a, c = o1["lvl_logits"][l], o2["lvl_logits"][l]
        m = ~torch.isnan(a[:, 0])
        assert torch.allclose(a[m], c[m], atol=1e-5)
    # ... while in FULL they do
    cfg, S = _setup("full")
    lay = build_layout(ps, w, S.vocab, S.opts)
    lay2 = build_layout(ps2, w, S.vocab, S.opts)
    with torch.no_grad():
        a = model.compute(collate([lay], S.tokenizer.pad_token_id, S.tc.L), keep=True)["lvl_logits"][0]
        c = model.compute(collate([lay2], S.tokenizer.pad_token_id, S.tc.L), keep=True)["lvl_logits"][0]
    assert not torch.allclose(a[1:], c[1:], atol=1e-5)


@needs_tiny
def test_loss_backward_and_desc():
    import torch
    from htpp.layout import build_layout, collate
    from htpp.model import HeadlessTPP
    cfg, S = _setup("full", ["train.gen=true", "train.lambda_d=0.5"])
    model = HeadlessTPP(cfg["model"], S.vocab, S.name_ids, "cpu").train()
    lays = [build_layout(S.pre["train"][w.seq], w, S.vocab, S.opts) for w in S.wins["train"][:4]]
    b = collate(lays, S.tokenizer.pad_token_id, S.tc.L)
    loss, parts = model.loss(b, {"time", "type", "desc", "sil"}, 0.5)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.new_out.grad is not None and torch.isfinite(model.new_out.grad).all()
    assert parts["nll_desc"] > 0


def test_capped_mean():
    rng = np.random.default_rng(4)
    tc = TimeCode.fit(rng.exponential(1.0, 20000), (8, 8), p_tail=0.02, resolution=1e-4)
    p = rng.dirichlet(np.ones(tc.n_fine + 1))
    m_full, _ = tc.mean_median(p[:-1], p[-1])
    m_big, _ = tc.mean_median(p[:-1], p[-1], cap=1e9)
    assert abs(m_full - m_big) < 1e-9
    # E[min(tau, c)] by numerical integration of the survival function
    c = 0.5 * tc.a_max
    m_cap, _ = tc.mean_median(p[:-1], p[-1], cap=c)
    xs = np.linspace(0, c, 200001)
    cdf_e = np.concatenate([[0], np.cumsum(p[:-1])])
    m = np.clip(np.searchsorted(tc.edges, xs, side="right") - 1, 0, tc.n_fine - 1)
    S = 1 - (cdf_e[m] + p[m] * (xs - tc.edges[m]) / tc.widths[m])
    assert abs(m_cap - np.trapezoid(S, xs)) < 1e-4 and m_cap <= c
