"""Neural TPP baselines with an exact likelihood: a history encoder (GRU as in
RMTPP/IFTPP, or a causal Transformer as in THP/SAHP) and a log-normal mixture for the
waiting time (Shchur et al., 2020) plus a categorical distribution for the type.
They score exactly the same events and quiet gaps as Headless-TPP, in the same time
unit, so the NLLs are directly comparable."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import EventSeq
from .evaluate import macro_f1
from .utils import Timer


class LNMTPP(nn.Module):
    def __init__(self, K, d=64, n_comp=8, encoder="gru", layers=2, heads=4, mean_log=0.0, std_log=1.0, dropout=0.1):
        super().__init__()
        self.K, self.C = K, n_comp
        self.emb = nn.Embedding(K, d)
        self.tproj = nn.Linear(2, d)
        self.encoder = encoder
        if encoder == "gru":
            self.enc = nn.GRU(d, d, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        else:
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 3 * n_comp + K)
        self.register_buffer("m", torch.tensor(float(mean_log)))
        self.register_buffer("s", torch.tensor(float(std_log)))
        self.d = d

    def _time_feats(self, tau, t):
        z = (torch.log(tau.clamp(min=1e-12)) - self.m) / self.s
        return torch.stack([z.clamp(-10, 10), torch.sin(t / (1.0 + t.abs().mean()))], -1)

    def hidden(self, tau, k, t, mask):
        x = self.emb(k) + self.tproj(self._time_feats(tau, t))
        if self.encoder == "gru":
            h, _ = self.enc(x)
        else:
            n = x.shape[1]
            causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
            pos = torch.arange(n, device=x.device)[:, None] / (10000 ** (torch.arange(0, self.d, 2, device=x.device)[None] / self.d))
            pe = torch.zeros(n, self.d, device=x.device)
            pe[:, 0::2] = torch.sin(pos)
            pe[:, 1::2] = torch.cos(pos)
            h = self.enc(x + pe[None], mask=causal, src_key_padding_mask=~mask)
        return h

    def params(self, h):
        o = self.head(h)
        C = self.C
        logw = o[..., :C].log_softmax(-1)
        mu = o[..., C:2 * C]
        logsig = o[..., 2 * C:3 * C].clamp(-5, 3)
        mark = o[..., 3 * C:].log_softmax(-1)
        return logw, mu, logsig, mark

    def log_density(self, tau, logw, mu, logsig):
        z = (torch.log(tau) - self.m) / self.s
        lz = -0.5 * ((z[..., None] - mu) / logsig.exp()) ** 2 - logsig - 0.5 * math.log(2 * math.pi)
        return torch.logsumexp(logw + lz, -1) - torch.log(self.s) - torch.log(tau)

    def log_survival(self, tau, logw, mu, logsig):
        z = (torch.log(tau) - self.m) / self.s
        u = (z[..., None] - mu) / logsig.exp()
        logsf = torch.special.log_ndtr(-u)
        return torch.logsumexp(logw + logsf, -1)

    def cdf(self, tau, logw, mu, logsig):
        return 1.0 - torch.exp(self.log_survival(tau, logw, mu, logsig))


def _pad(seqs: list[EventSeq], tau_min: float):
    n = max(len(s) for s in seqs)
    B = len(seqs)
    tau = torch.ones(B, n)
    t = torch.zeros(B, n)
    k = torch.zeros(B, n, dtype=torch.long)
    mask = torch.zeros(B, n, dtype=torch.bool)
    gap = torch.full((B,), float("nan"))
    hist = torch.zeros(B, n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        hist[i, : max(1, int(getattr(s, "n_hist", 0) or 0))] = True
        m = len(s)
        tau[i, :m] = torch.tensor(np.maximum(s.tau, tau_min))
        t[i, :m] = torch.tensor(s.t - s.t[0])
        k[i, :m] = torch.tensor(s.k)
        mask[i, :m] = True
        if s.t_end is not None and s.t_end > s.t[-1]:
            gap[i] = max(s.t_end - s.t[-1], tau_min)
    return tau, t, k, mask & ~hist, gap, mask


def _seq_terms(model, tau, t, k, scored, gap, mask):
    """Per-event NLL parts of events 1..n-1 and per-sequence silence NLL."""
    h = model.hidden(tau, k, t, mask)
    logw, mu, logsig, mark = model.params(h)
    lp_t = model.log_density(tau[:, 1:], logw[:, :-1], mu[:, :-1], logsig[:, :-1])
    lp_k = mark[:, :-1].gather(-1, k[:, 1:, None]).squeeze(-1)
    valid = scored[:, 1:]
    last = mask.sum(1) - 1
    bi = torch.arange(tau.shape[0], device=tau.device)
    has_gap = ~torch.isnan(gap)
    g = torch.where(has_gap, gap, torch.ones_like(gap))
    lS = model.log_survival(g, logw[bi, last], mu[bi, last], logsig[bi, last])
    lS = torch.where(has_gap, lS, torch.zeros_like(lS))
    return lp_t, lp_k, valid, lS, (logw, mu, logsig, mark)


def run_baseline(cfg: dict, splits: dict, K: int, out_dir: str | Path, log, true_nll=None) -> dict:
    bcfg = cfg.get("baseline", {})
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["train"].get("seed", 1)))
    rng = np.random.default_rng(int(cfg["train"].get("seed", 1)))
    train, val, test = splits["train"], splits["val"], splits["test"]
    taus = np.concatenate([s.tau[1:] for s in train if len(s) > 1])
    res = cfg.get("timecode", {}).get("time_resolution") or 1e-6 * float(np.median(taus[taus > 0]))
    tau_min = 0.5 * float(res)
    cap = float(max(float(np.max(s.tau[1:])) for s in train if len(s) > 1))
    lt = np.log(np.maximum(taus, tau_min))
    model = LNMTPP(K, d=int(bcfg.get("d", 64)), n_comp=int(bcfg.get("n_comp", 8)), encoder=bcfg.get("encoder", "gru"),
                   layers=int(bcfg.get("layers", 2)), heads=int(bcfg.get("heads", 4)), mean_log=lt.mean(),
                   std_log=lt.std() + 1e-6).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(bcfg.get("lr", 1e-3)))
    bs = int(bcfg.get("batch_size", 64))
    timer = Timer()

    def batches(seqs, shuffle):
        idx = np.arange(len(seqs))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [seqs[j] for j in idx[i:i + bs]]

    def nll_of(seqs):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in batches(seqs, False):
                tau, t, k, sc, gap, mask = [x.to(dev) for x in _pad(b, tau_min)]
                lp_t, lp_k, valid, lS, _ = _seq_terms(model, tau, t, k, sc, gap, mask)
                tot -= float(((lp_t + lp_k) * valid).sum() + lS.sum())
                n += int(valid.sum())
        model.train()
        return tot / max(n, 1)

    best, bad, best_state = float("inf"), 0, None
    for ep in range(int(bcfg.get("epochs", 100))):
        for b in batches(train, True):
            tau, t, k, sc, gap, mask = [x.to(dev) for x in _pad(b, tau_min)]
            lp_t, lp_k, valid, lS, _ = _seq_terms(model, tau, t, k, sc, gap, mask)
            loss = -(((lp_t + lp_k) * valid).sum() + lS.sum()) / valid.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        v = nll_of(val)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0:
            log.info(f"baseline epoch {ep} val nll {v:.4f} best {best:.4f}")
        if bad >= int(bcfg.get("patience", 10)):
            break
    model.load_state_dict(best_state)
    # ----- test metrics
    model.eval()
    rows, gaps = [], []
    deltas = [float(d) for d in (cfg.get("eval", {}).get("interval_deltas") or [])]   # see htpp/interval.py
    with torch.no_grad():
        for b in batches(test, False):
            tau, t, k, sc, gap, mask = [x.to(dev) for x in _pad(b, tau_min)]
            lp_t, lp_k, valid, lS, (logw, mu, logsig, mark) = _seq_terms(model, tau, t, k, sc, gap, mask)
            pred_k = mark[:, :-1].argmax(-1)
            # PIT, mean and median of the predictive waiting-time distribution
            lw, m_, ls = logw[:, :-1], mu[:, :-1], logsig[:, :-1]
            pit = model.cdf(tau[:, 1:], lw, m_, ls)
            # E[min(tau, cap)] of the log-normal mixture (the same capped mean as for Headless-TPP)
            mu_l = model.m + model.s * m_
            sg = model.s * ls.exp()
            lc = math.log(cap)
            t1 = torch.exp(mu_l + 0.5 * sg ** 2 + torch.special.log_ndtr((lc - mu_l - sg ** 2) / sg))
            t2 = cap * torch.exp(torch.special.log_ndtr(-(lc - mu_l) / sg))
            mean = (lw.exp() * (t1 + t2)).sum(-1)
            lo = torch.full_like(mean, -30.0)
            hi = torch.full_like(mean, 30.0)
            for _ in range(60):   # bisection on log tau for the median
                mid = (lo + hi) / 2
                c = model.cdf(torch.exp(mid), lw, m_, ls)
                lo = torch.where(c < 0.5, mid, lo)
                hi = torch.where(c < 0.5, hi, mid)
            med = torch.exp((lo + hi) / 2)
            ints = []
            if deltas:   # probability of the recorded interval of waiting times (double precision)
                raw = torch.zeros(tau.shape, dtype=torch.float64, device=tau.device)
                for i, s in enumerate(b):
                    raw[i, : len(s)] = torch.tensor(np.asarray(s.tau, float), dtype=torch.float64)
                r1 = raw[:, 1:]
                lwd, mud, lsd = lw.double(), m_.double(), ls.double()
                for dl in deltas:
                    small = r1 < dl / 2
                    lo_ = torch.where(small, torch.zeros_like(r1), r1 - dl / 2)
                    hi_ = torch.where(small, torch.full_like(r1, dl), r1 + dl / 2)
                    s_lo = model.log_survival(lo_, lwd, mud, lsd)
                    s_hi = model.log_survival(hi_, lwd, mud, lsd)
                    lp = s_lo + torch.log(-torch.expm1((s_hi - s_lo).clamp(max=-1e-15)))
                    ints.append(-(lp - torch.log(hi_ - lo_)))
            for i, s in enumerate(b):
                for j in range(max(1, int(s.n_hist)), len(s)):
                    rows.append({"seq": s.sid, "idx": j, "k": int(s.k[j]), "prev_k": int(s.k[j - 1]), "hist_len": j,
                                 "tau": float(s.tau[j]), "nll_time": float(-lp_t[i, j - 1]), "nll_type": float(-lp_k[i, j - 1]),
                                 "pred_k": int(pred_k[i, j - 1]), "pit": float(pit[i, j - 1]),
                                 "pred_mean": float(mean[i, j - 1]), "pred_median": float(med[i, j - 1]),
                                 **{f"nll_time_int{di}": float(x[i, j - 1]) for di, x in enumerate(ints)}})
                if not math.isnan(float(gap[i])):
                    gaps.append({"seq": s.sid, "gap": float(gap[i]), "nll_sil": float(-lS[i]), "true_nll": s.true_nll})
    df = pd.DataFrame(rows)
    gdf = pd.DataFrame(gaps)
    n = len(df)
    m = {"n_events": n, "n_gaps": len(gdf), "nll_time": float(df.nll_time.sum() / n), "nll_type": float(df.nll_type.sum() / n),
         "nll_sil": float(gdf.nll_sil.sum() / n) if len(gdf) else 0.0}
    m["nll"] = m["nll_time"] + m["nll_type"] + m["nll_sil"]
    m["type_acc"] = float((df.k == df.pred_k).mean())
    m["type_macro_f1"] = macro_f1(df.k.values, df.pred_k.values)
    pit = np.sort(df.pit.clip(0, 1).values)
    m["pit_ks"] = float(np.max(np.abs(pit - (np.arange(1, n + 1) - 0.5) / n)))
    m["pit_ece"] = float(np.mean(np.abs(np.histogram(pit, bins=10, range=(0, 1))[0] / n - 0.1)))
    m["rmse_mean"] = float(np.sqrt(np.mean((df.pred_mean - df.tau) ** 2)))
    m["mae_median"] = float(np.mean(np.abs(df.pred_median - df.tau)))
    m["mae_log"] = float(np.mean(np.abs(np.log1p(df.pred_median) - np.log1p(df.tau))))
    tn = [s.true_nll for s in test]
    if all(x is not None for x in tn):
        m["nll_true"] = float(np.sum(tn) / n)
        m["gap_to_true"] = m["nll"] - m["nll_true"]
    if deltas:
        m["interval"] = {}
        for di, dl in enumerate(deltas):
            it = float(df[f"nll_time_int{di}"].sum() / n)
            m["interval"][repr(dl)] = {"delta": dl, "nll_time": it, "nll": it + m["nll_type"] + m["nll_sil"]}
    m["pred_cap"] = cap
    m["params"] = int(sum(p.numel() for p in model.parameters()))
    m["train_seconds"] = timer()
    out_dir = Path(out_dir)
    df.to_csv(out_dir / "test_events.csv.gz", index=False)
    gdf.to_csv(out_dir / "test_gaps.csv.gz", index=False)
    return m
