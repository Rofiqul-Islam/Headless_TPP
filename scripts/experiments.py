#!/usr/bin/env python
"""Run all experiments of the paper on a pool of GPUs.

  python scripts/experiments.py rq1 --gpus 0,1,2,3            # RQ1: E1a (Hawkes), E1b (benchmarks), E1c (silence)
  python scripts/experiments.py rq3 --gpus 0,1,2,3            # RQ3: six inputs x two budgets, NAMED-OUT, FULL+GEN
  python scripts/experiments.py all --gpus 0,1 --dry_run      # print the commands only

Each job is one call of scripts/run.py and writes to its own folder under runs/. Jobs
whose summary.json already exists are skipped, so the command can be re-run after an
interruption. Use --jobs_per_gpu 2 for the small baselines or the 0.5B model.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

NO_SIL = "train.stages=[{name: grammar, steps: 200, losses: [time, type], train: new_tokens, synthetic_frac: 0.5}, " \
         "{name: next_event, epochs: 3, losses: [time, type]}]"


EXTRA_CONFIGS: list[str] = []


def job(config, out, sets):
    cfgs = [str(ROOT / config)] + EXTRA_CONFIGS
    return {"cmd": [PY, str(ROOT / "scripts" / "run.py"), "--config", *cfgs, "--set", f"out_dir={out}", *sets],
            "out": out}


def rq1_jobs(a):
    J = []
    M = [f"model.name={a.model}"]
    small = [f"model.name={a.small_model}", "model.pretrained=false", "train.lr=3.0e-4"]
    for s in a.seeds:
        seed = [f"train.seed={s}"]
        # E1a: synthetic Hawkes with a known likelihood -- does the fine-tuned LLM reach the true NLL?
        d = [f"data.dir={a.data}/hawkes", "experiment=rq1_hawkes"]
        for var, extra in (("hier8x8x8", []), ("hier32x16", ["timecode.branching=[32,16]"]),
                           ("flat512", ["timecode.branching=[512]", "eval.predict_depth=1"]),
                           ("digits", ["timecode.kind=digits"])):
            J.append(job("configs/rq1.yaml", f"{a.runs}/rq1_hawkes/{var}/s{s}", M + d + seed + [f"variant={var}"] + extra))
        for enc in ("gru", "transformer"):
            J.append(job("configs/rq1.yaml", f"{a.runs}/rq1_hawkes/lnm_{enc}/s{s}",
                         d + seed + ["method=baseline", f"baseline.encoder={enc}", f"variant=lnm_{enc}"]))
        # E1b: benchmarks
        for ds in a.datasets:
            d = [f"data.dir={a.data}/{ds}", "experiment=rq1_bench"]
            J.append(job("configs/rq1.yaml", f"{a.runs}/rq1_bench/{ds}/headless/s{s}", M + d + seed + ["variant=headless"]))
            J.append(job("configs/rq1.yaml", f"{a.runs}/rq1_bench/{ds}/digits/s{s}",
                         M + d + seed + ["variant=digits", "timecode.kind=digits"]))
            J.append(job("configs/rq1.yaml", f"{a.runs}/rq1_bench/{ds}/scratch/s{s}", small + d + seed + ["variant=scratch"]))
            for enc in ("gru", "transformer"):
                J.append(job("configs/rq1.yaml", f"{a.runs}/rq1_bench/{ds}/lnm_{enc}/s{s}",
                             d + seed + ["method=baseline", f"baseline.encoder={enc}", f"variant=lnm_{enc}"]))
        # E1c: learning from silence -- the same model with and without the survival term
        for ds in a.silence_datasets:
            d = [f"data.dir={a.data}/{ds}", "experiment=rq1_silence"]
            J.append(job("configs/rq1.yaml", f"{a.runs}/rq1_silence/{ds}/with_sil/s{s}", M + d + seed + ["variant=with_sil"]))
            J.append(job("configs/rq1.yaml", f"{a.runs}/rq1_silence/{ds}/no_sil/s{s}", M + d + seed + ["variant=no_sil", NO_SIL]))
    return J


def rq3_jobs(a):
    J = []
    base = [f"model.name={a.model}", f"data.dir={a.data}/{a.amazon}", "experiment=rq3"]
    for s in a.seeds:
        seed = [f"train.seed={s}"]
        for budget in ("event", "token"):
            for cond in ("no_text", "names", "scrambled", "full", "shuffled", "text_only"):
                J.append(job("configs/rq3.yaml", f"{a.runs}/rq3/{budget}/{cond}/s{s}",
                             base + seed + [f"data.condition={cond}", f"data.budget={budget}", f"variant={cond}"]))
        for cond in ("no_text", "full"):   # NAMED-OUT: type tokens start from the name's embedding
            J.append(job("configs/rq3.yaml", f"{a.runs}/rq3/named_out/{cond}/s{s}",
                         base + seed + [f"data.condition={cond}", "model.mark_init=name", f"variant=named_out_{cond}"]))
        for frac in a.train_fracs:         # training size: keep a share of the training users
            for cond in ("no_text", "full"):
                J.append(job("configs/rq3.yaml", f"{a.runs}/rq3/train_frac{frac}/{cond}/s{s}",
                             base + seed + [f"data.condition={cond}", f"data.train_frac={frac}", f"variant={cond}_frac{frac}"]))
        for lam in a.lambdas:              # FULL+GEN: the model also writes the next review
            J.append(job("configs/rq3.yaml", f"{a.runs}/rq3/gen/lambda{lam}/s{s}",
                         base + seed + ["data.condition=full", "train.gen=true", f"train.lambda_d={lam}", f"variant=gen{lam}"]))
    return J


def run_pool(jobs, gpus, per_gpu, dry, log_dir):
    todo = [j for j in jobs if not (Path(j["out"]) / "summary.json").exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run on GPUs {gpus} x{per_gpu}")
    if dry:
        for j in todo:
            print(" ".join(f"'{c}'" if " " in c else c for c in j["cmd"]))
        return
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}
    failed = []
    while todo or running:
        for i, g in enumerate(slots):
            if i not in running and todo:
                j = todo.pop(0)
                Path(j["out"]).mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                f = open(Path(j["out"]) / "stdout.txt", "w", encoding="utf-8")
                running[i] = (subprocess.Popen(j["cmd"], env=env, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT), j, f, time.time())
                print(f"[gpu {g}] start {j['out']}", flush=True)
        for i in list(running):
            p, j, f, t0 = running[i]
            if p.poll() is not None:
                f.close()
                status = "done" if p.returncode == 0 else f"FAILED ({p.returncode}) see {j['out']}/stdout.txt"
                if p.returncode != 0:
                    failed.append(j["out"])
                print(f"[gpu {slots[i]}] {status} {j['out']} in {(time.time() - t0) / 60:.1f} min", flush=True)
                del running[i]
        time.sleep(5)
    print(f"finished; {len(failed)} failed" + ("".join(f"\n  {x}" for x in failed)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["rq1", "rq3", "all"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--jobs_per_gpu", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--small_model", default="Qwen/Qwen2.5-0.5B", help="architecture for the no-pre-training ablation")
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--datasets", nargs="+", default=["retweet", "stackoverflow", "taobao", "amazon"])
    ap.add_argument("--silence_datasets", nargs="+", default=["hawkes_short", "taobao_cut"])
    ap.add_argument("--amazon", default="amazon_phones")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 1.0])
    ap.add_argument("--train_fracs", type=float, nargs="*", default=[0.1, 0.3])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--extra_config", nargs="*", default=[], help="config files applied after the experiment config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides added to every job")
    a = ap.parse_args()
    a.data = str(Path(a.data).resolve())
    a.runs = str(Path(a.runs).resolve())
    EXTRA_CONFIGS.extend(str(Path(c).resolve()) for c in a.extra_config)
    jobs = []
    if a.which in ("rq1", "all"):
        jobs += rq1_jobs(a)
    if a.which in ("rq3", "all"):
        jobs += rq3_jobs(a)
    for j in jobs:
        j["cmd"] += a.set
    run_pool(jobs, [g.strip() for g in a.gpus.split(",") if g.strip()], a.jobs_per_gpu, a.dry_run, a.runs)


if __name__ == "__main__":
    main()
