"""Headless-TPP model: a causal LLM (LoRA-adapted) whose own output layer scores the
time tokens, the type tokens and (optionally) the description words. There is no
separate time head: new tokens get an input row and an output row, exactly like
ordinary words, and every probability is a softmax over the tokens that the event
grammar allows at that position (rule R2)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .layout import EventVocab
from .timecode import log_survival_torch

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class HeadlessTPP(nn.Module):
    def __init__(self, mcfg: dict, vocab: EventVocab, name_token_ids: list | None = None, device="cuda"):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM
        self.vocab = vocab
        self.device_ = torch.device(device)
        self.dtype = _dtype(mcfg.get("dtype", "bfloat16")) if self.device_.type == "cuda" else \
            _dtype(mcfg.get("cpu_dtype", "float32"))
        attn = mcfg.get("attn_implementation", "sdpa")
        if mcfg.get("pretrained", True):
            lm = AutoModelForCausalLM.from_pretrained(mcfg["name"], torch_dtype=self.dtype, attn_implementation=attn)
        else:   # same architecture, random weights ("no language pre-training" ablation)
            conf = AutoConfig.from_pretrained(mcfg["name"])
            conf._attn_implementation = attn
            for k, v in (mcfg.get("scratch_overrides") or {}).items():
                setattr(conf, k, v)
            lm = AutoModelForCausalLM.from_config(conf, torch_dtype=self.dtype)
        lm.config.use_cache = False
        self.use_lora = bool(mcfg.get("pretrained", True)) and int(mcfg.get("lora", {}).get("r", 16)) > 0
        if self.use_lora:
            from peft import LoraConfig, get_peft_model
            lc = mcfg.get("lora", {})
            lm = get_peft_model(lm, LoraConfig(r=int(lc.get("r", 16)), lora_alpha=int(lc.get("alpha", 32)),
                                               lora_dropout=float(lc.get("dropout", 0.05)),
                                               target_modules=lc.get("targets", LORA_TARGETS),
                                               task_type="CAUSAL_LM"))
            base = lm.get_base_model()
        else:
            base = lm
        if mcfg.get("gradient_checkpointing", True):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lm = lm
        self.inner = base.get_decoder() if hasattr(base, "get_decoder") else base.model
        self.embed = base.get_input_embeddings()
        self.out_w = base.get_output_embeddings().weight
        self.tied = self.out_w.data_ptr() == self.embed.weight.data_ptr()
        self.V = vocab.V
        assert self.embed.num_embeddings == self.V, "vocab.V must equal the base embedding rows"
        d = self.embed.embedding_dim
        n_new = vocab.n_new
        # ---- initialise the new rows (input and output) --------------------------------
        with torch.no_grad():
            E = self.embed.weight[: vocab.V_tok].float()
            mu, sd = E.mean(0), E.std(0)
            g = torch.Generator().manual_seed(1234)
            new_in = mu[None].cpu() + 0.1 * sd[None].cpu() * torch.randn(n_new, d, generator=g)
            O = self.out_w[: vocab.V_tok].float()
            omu, osd = O.mean(0), O.std(0)
            new_out = omu[None].cpu() + 0.1 * osd[None].cpu() * torch.randn(n_new, d, generator=g)
            if name_token_ids is not None:   # NAMED-OUT: type tokens start from the name's embedding
                for j, toks in enumerate(name_token_ids):
                    if not toks:
                        continue
                    r = vocab.mark_ids[j] - vocab.V
                    t = torch.tensor(toks)
                    new_in[r] = self.embed.weight[t].float().mean(0).cpu()
                    new_out[r] = self.out_w[t].float().mean(0).cpu()
        self.new_in = nn.Parameter(new_in)
        self.new_out = nn.Parameter(new_out)
        if not self.use_lora:
            for p in self.lm.parameters():
                p.requires_grad_(not bool(mcfg.get("pretrained", True)))
        self.level_ids = [torch.tensor(x, dtype=torch.long) for x in vocab.level_ids]
        self.mark_ids = torch.tensor(vocab.mark_ids, dtype=torch.long)
        self.to(self.device_)
        self.new_in.data = self.new_in.data.float()
        self.new_out.data = self.new_out.data.float()
        self.level_ids = [x.to(self.device_) for x in self.level_ids]
        self.mark_ids = self.mark_ids.to(self.device_)

    # ------------------------------------------------------------------ params
    def lora_parameters(self):
        return [p for n, p in self.lm.named_parameters() if p.requires_grad]

    def new_token_parameters(self):
        return [self.new_in, self.new_out]

    def set_lora_trainable(self, flag: bool):
        for n, p in self.lm.named_parameters():
            if self.use_lora:
                if "lora_" in n:
                    p.requires_grad_(flag)
            else:
                p.requires_grad_(flag)

    def trainable_state(self) -> dict:
        st = {"new_in": self.new_in.detach().cpu(), "new_out": self.new_out.detach().cpu()}
        if self.use_lora:
            from peft import get_peft_model_state_dict
            st["lora"] = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.lm).items()}
        else:
            st["lm"] = {k: v.detach().cpu() for k, v in self.lm.state_dict().items()}
        return st

    def load_trainable_state(self, st: dict):
        self.new_in.data.copy_(st["new_in"].to(self.device_))
        self.new_out.data.copy_(st["new_out"].to(self.device_))
        if self.use_lora:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.lm, st["lora"])
        else:
            self.lm.load_state_dict(st["lm"])

    # ----------------------------------------------------------------- forward
    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        is_new = ids >= self.V
        x = self.embed(ids.clamp(max=self.V - 1))
        if bool(is_new.any()):
            nx = self.new_in[(ids - self.V).clamp(min=0, max=self.vocab.n_new - 1)].to(x.dtype)
            x = torch.where(is_new.unsqueeze(-1), nx, x)
        return x

    def hidden(self, batch: dict) -> torch.Tensor:
        dev = self.device_
        ids = batch["input_ids"].to(dev)
        x = self.embed_ids(ids)
        if batch.get("mask4d") is not None:
            attn = batch["mask4d"].to(dev, dtype=x.dtype)
        else:
            attn = batch["attention_mask"].to(dev)
        out = self.inner(inputs_embeds=x, attention_mask=attn, position_ids=batch["position_ids"].to(dev),
                         use_cache=False)
        return out.last_hidden_state

    def out_rows(self, ids: torch.Tensor) -> torch.Tensor:
        base = ids < self.V
        Wb = self.out_w[ids.clamp(max=self.V - 1)].float()
        Wn = self.new_out[(ids - self.V).clamp(min=0)]
        return torch.where(base.unsqueeze(-1), Wb, Wn)

    def level_logits(self, h: torch.Tensor, level: int) -> torch.Tensor:
        return h.float() @ self.out_rows(self.level_ids[level]).T

    def mark_logits(self, h: torch.Tensor) -> torch.Tensor:
        return h.float() @ self.out_rows(self.mark_ids).T

    def desc_logprobs(self, h: torch.Tensor, tgt: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """log P(word) over the ordinary vocabulary plus </desc> (grammar inside a description)."""
        Vt = self.vocab.V_tok
        W = self.out_w[:Vt]
        close = self.new_out[self.vocab.desc_close - self.V]

        def fn(hc, tc):
            logits = (hc.to(W.dtype) @ W.T).float()
            cl = hc.float() @ close
            full = torch.cat([logits, cl[:, None]], 1)
            lse = torch.logsumexp(full, 1)
            idx = torch.where(tc >= 0, tc, torch.full_like(tc, Vt))
            return full.gather(1, idx[:, None]).squeeze(1) - lse

        outs = []
        for i in range(0, h.shape[0], chunk):
            hc, tc = h[i:i + chunk], tgt[i:i + chunk]
            if self.training:
                outs.append(checkpoint(fn, hc, tc, use_reentrant=False))
            else:
                outs.append(fn(hc, tc))
        return torch.cat(outs) if outs else h.new_zeros(0)

    def compute(self, batch: dict, use_desc: bool = False, temps: dict | None = None, keep: bool = False) -> dict:
        """Per-event log-probabilities (and per-gap log-survival) for one batch."""
        dev = self.device_
        h = self.hidden(batch)
        n_ev = batch["n_events"]
        L = len(self.level_ids)
        out = {"lp_time": torch.zeros(n_ev, device=dev), "lp_type": torch.zeros(n_ev, device=dev),
               "lp_desc": torch.zeros(n_ev, device=dev)}
        tT = (temps or {}).get("time", [1.0] * L)
        tK = (temps or {}).get("mark", 1.0)
        if keep:
            out["lvl_logits"] = [torch.full((n_ev, len(self.level_ids[lv])), float("nan")) for lv in range(L)]
        for lv, rd in enumerate(batch["time_reads"]):
            if rd is None:
                continue
            hs = h[rd["b"].to(dev), rd["pos"].to(dev)]
            lg = self.level_logits(hs, lv)
            if keep:
                out["lvl_logits"][lv][rd["ev"]] = lg.detach().float().cpu()
            lp = (lg / tT[lv]).log_softmax(-1)
            out["lp_time"].index_add_(0, rd["ev"].to(dev), lp.gather(1, rd["tgt"].to(dev)[:, None]).squeeze(1))
        rd = batch["mark_reads"]
        if rd is not None:
            hs = h[rd["b"].to(dev), rd["pos"].to(dev)]
            lg = self.mark_logits(hs)
            if keep:
                ml = torch.full((n_ev, lg.shape[1]), float("nan"))
                ml[rd["ev"]] = lg.detach().float().cpu()
                out["mark_logits"] = ml
            lp = (lg / tK).log_softmax(-1)
            out["lp_type"].index_add_(0, rd["ev"].to(dev), lp.gather(1, rd["tgt"].to(dev)[:, None]).squeeze(1))
        rd = batch.get("desc_reads")
        if use_desc and rd is not None:
            hs = h[rd["b"].to(dev), rd["pos"].to(dev)]
            lp = self.desc_logprobs(hs, rd["tgt"].to(dev))
            out["lp_desc"].index_add_(0, rd["ev"].to(dev), lp)
        if batch["gaps"]:
            G = len(batch["gaps"])
            lvl = []
            keep_g = [] if keep else None
            for lv in range(L):
                lg_full = torch.full((G, len(self.level_ids[lv])), 0.0, device=dev)
                rd = batch["gap_reads"][lv]
                if rd is not None:
                    hs = h[rd["b"].to(dev), rd["pos"].to(dev)]
                    lg = self.level_logits(hs, lv)
                    lg_full = lg_full.index_copy(0, rd["row"].to(dev), lg)
                if keep:
                    keep_g.append(lg_full.detach().float().cpu())
                lvl.append((lg_full / tT[lv]).log_softmax(-1))
            out["logS_gap"] = log_survival_torch(lvl, batch["gap_path"].to(dev), batch["gap_plen"].to(dev),
                                                 batch["gap_log_r"].to(dev))
            if keep:
                out["gap_lvl_logits"] = keep_g
        else:
            out["logS_gap"] = torch.zeros(0, device=dev)
        if keep and batch["branches"]:
            br = []
            for (ev, b, rows) in batch["branches"]:
                per = []
                for lv in sorted(rows):
                    hs = h[b, torch.tensor(rows[lv], device=dev)]
                    per.append(self.level_logits(hs, lv).detach().float().cpu())
                br.append((ev, per))
            out["branch_logits"] = br
        return out

    def loss(self, batch: dict, losses: set, lambda_d: float = 0.0) -> tuple[torch.Tensor, dict]:
        use_desc = "desc" in losses and lambda_d > 0
        o = self.compute(batch, use_desc=use_desc)
        n = max(batch["n_events"], 1)
        tot = torch.zeros((), device=self.device_)
        log_u = batch["ev_log_u"].to(self.device_)
        parts = {}
        if "time" in losses:
            tot = tot - o["lp_time"].sum()
        if "type" in losses:
            tot = tot - o["lp_type"].sum()
        if use_desc:
            tot = tot - lambda_d * o["lp_desc"].sum()
        if "sil" in losses and o["logS_gap"].numel():
            tot = tot - o["logS_gap"].sum()
        parts["nll_time"] = float(-(o["lp_time"].sum() + log_u.sum()).item()) / n
        parts["nll_type"] = float(-o["lp_type"].sum().item()) / n
        parts["nll_sil"] = float(-o["logS_gap"].sum().item()) / n if o["logS_gap"].numel() else 0.0
        parts["nll_desc"] = float(-o["lp_desc"].sum().item()) / n
        return tot / n, parts
