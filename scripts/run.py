#!/usr/bin/env python
"""Train and evaluate one model (Headless-TPP or a baseline) for one configuration.

    python scripts/run.py --config configs/rq3_amazon.yaml \
        --set data.condition=full train.seed=2 out_dir=runs/rq3/full_event_s2

Writes to out_dir: config.json, log.txt, best.pt, summary.json, test_events.csv.gz,
test_gaps.csv.gz (and generations.jsonl for FULL+GEN). A run whose summary.json
already exists is skipped (use --set overwrite=true to redo it).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from htpp.utils import env_info, get_logger, load_config, save_json, set_seed, write_jsonl  # noqa: E402


def tags_of(cfg: dict) -> dict:
    d = cfg["data"]
    method = cfg.get("method", "headless")
    if method == "baseline":
        model = "baseline-" + cfg.get("baseline", {}).get("encoder", "gru")
    else:
        model = Path(cfg["model"]["name"]).name + ("" if cfg["model"].get("pretrained", True) else "-scratch")
    tc = cfg.get("timecode", {})
    code = "digits" if tc.get("kind") == "digits" else "hier" + "x".join(str(b) for b in tc.get("branching", []))
    return {"experiment": cfg.get("experiment", "debug"), "variant": cfg.get("variant", "base"),
            "dataset": Path(d["dir"]).name, "model": model, "method": method,
            "condition": d.get("condition", "no_text"), "budget": d.get("budget", "event"),
            "seed": int(cfg["train"].get("seed", 1)), "gen": bool(cfg["train"].get("gen", False)),
            "lambda_d": float(cfg["train"].get("lambda_d", 0.0)), "train_frac": float(d.get("train_frac") or 1.0), "mark_init": cfg.get("model", {}).get("mark_init", "mean"),
            "timecode": code, "silence": "sil" in set(cfg["train"]["stages"][-1]["losses"]) if method != "baseline" else True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.sub=value")
    ap.add_argument("--eval_only", action="store_true", help="load out_dir/best.pt instead of training")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg["out_dir"])
    if (out / "summary.json").exists() and not cfg.get("overwrite", False):
        print(f"skip {out} (summary.json exists)")
        return
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out)
    seed = int(cfg["train"].get("seed", 1))
    set_seed(seed)
    save_json(cfg, out / "config.json")
    tags = tags_of(cfg)
    log.info(f"run {out} | tags {tags}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.get("method", "headless") == "baseline":
        from htpp.baselines import run_baseline
        from htpp.data import load_meta, load_split
        meta = load_meta(cfg["data"]["dir"])
        splits = {s: load_split(cfg["data"]["dir"], s, cfg["data"].get(f"max_{s}_seqs")) for s in ("train", "val", "test")}
        cfg.setdefault("timecode", {}).setdefault("time_resolution", meta.get("time_resolution"))
        m = run_baseline(cfg, splits, int(meta["K"]), out, log)
        summary = {"tags": tags, "metrics": m, "metrics_raw": m, "env": env_info(), "config": cfg}
        save_json(summary, out / "summary.json")
        log.info(f"test {m}")
        return

    from htpp.evaluate import collect, compute_metrics, fit_temperatures, text_features
    from htpp.model import HeadlessTPP
    from htpp.pipeline import pick_events, setup
    from htpp.train import train

    S = setup(cfg, log)
    name_ids = S.name_ids if cfg["model"].get("mark_init", "mean") == "name" else None
    model = HeadlessTPP(cfg["model"], S.vocab, name_ids, device)
    tr = {}
    if args.eval_only:
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        model.load_trainable_state(ck["state"])
    else:
        tr = train(model, S, log, out)
    ecfg = cfg.get("eval", {})
    gen = bool(cfg["train"].get("gen", False)) and float(cfg["train"].get("lambda_d", 0)) > 0
    # stage 4: temperature calibration on validation data
    temps = None
    if ecfg.get("calibrate", True):
        rec_val = collect(model, S, "val", use_desc=False)
        temps = fit_temperatures(rec_val, S.tc.L, marks=bool(ecfg.get("calibrate_marks", True)))
        log.info(f"calibration temperatures {temps}")
    depth = int(ecfg.get("predict_depth", S.tc.L))
    n_pred = int(ecfg.get("predict_max_events", 3000))
    branch_events = pick_events(S.pre["test"], n_pred, 0) if depth > 0 and n_pred > 0 else None
    rec = collect(model, S, "test", branch_events, depth, use_desc=gen)
    cap = float(max(float(np.max(s.tau[1:])) for s in S.raw["train"] if len(s) > 1))   # predictions use E[min(tau, cap)]
    m_raw, _, _ = compute_metrics(rec, S, None, depth, cap)
    m_cal, df, gdf = compute_metrics(rec, S, temps, depth, cap)
    m_cal["pred_cap"] = cap

    def add_text(df, pre):
        if S.meta.get("has_text"):
            tf = text_features(model, pre)
            df["sim_prev"] = [tf.get((a, b), (np.nan, np.nan))[0] for a, b in zip(df.seq, df.idx)]
            df["prev_len"] = [tf.get((a, b), (np.nan, np.nan))[1] for a, b in zip(df.seq, df.idx)]
        return df

    add_text(df, S.pre["test"]).to_csv(out / "test_events.csv.gz", index=False)
    gdf.to_csv(out / "test_gaps.csv.gz", index=False)
    m_unseen = None
    if "test_unseen" in S.pre and S.wins.get("test_unseen"):   # users never seen in training (cold start)
        rec_u = collect(model, S, "test_unseen", None, 0, use_desc=gen)
        m_unseen, df_u, gdf_u = compute_metrics(rec_u, S, temps, 0)
        add_text(df_u, S.pre["test_unseen"]).to_csv(out / "test_unseen_events.csv.gz", index=False)
        gdf_u.to_csv(out / "test_unseen_gaps.csv.gz", index=False)
    summary = {"tags": tags, "metrics": m_cal, "metrics_raw": m_raw, "metrics_unseen": m_unseen, "temps": temps,
               "timecode": S.tc.to_dict() if S.tc.kind == "hier" else {"kind": "digits", "branching": S.tc.branching},
               "train": {k: v for k, v in tr.items() if k != "history"}, "history": tr.get("history"),
               "env": env_info(), "config": cfg}
    if gen and int(ecfg.get("gen_max_events", 0)) > 0:
        if S.opts.condition == "text_only":
            raise ValueError("description writing is defined for the FULL input")
        from htpp.generate import generate_eval
        g = generate_eval(model, S, int(ecfg["gen_max_events"]), float(ecfg.get("top_p", 0.9)), seed)
        summary["generation"] = g["summary"]
        write_jsonl(g["rows"], out / "generations.jsonl")
    save_json(summary, out / "summary.json")
    log.info("test (calibrated): " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m_cal.items()))


if __name__ == "__main__":
    main()
