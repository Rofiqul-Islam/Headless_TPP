# Results

Results of the runs reported in the depth-exam response *LLMs as Temporal Point Processes:
Token Likelihoods and Event Text* (Qwen2.5-1.5B with LoRA; see the main README for the code).

| Folder | Contents |
|---|---|
| `tables/` | The LaTeX tables of the paper as written by `scripts/make_tables.py`, `results.json` (every number behind the tables, with intervals) and the figure `rq3_history.pdf` |
| `run_summaries/` | One folder per finished run: `summary.json` (test metrics after and before calibration, calibration temperatures, time code, training history), `config.json` (the exact settings), `log.txt` and `stdout.txt` |

Not included: model checkpoints (`best.pt`, several GB), the per-event files
(`test_events.csv.gz`, `test_gaps.csv.gz`) and generated reviews (`generations.jsonl`;
reviews can contain personal details). Paths of the training machine are shortened to
`<project>` and `<home>`.

## What was run

| Experiment | Paper table | Finished runs | Planned |
|---|---|---|---|
| RQ1, E1a: synthetic Hawkes data with a known likelihood | `rq1_hawkes.tex` | 18 (3 seeds) | 18 |
| RQ1, E1b: Retweet, StackOverflow, Taobao, Amazon (EasyTPP) | `rq1_bench.tex` | 41 (seeds 1–2; seed 3 only for Retweet Headless-TPP) | 60 |
| RQ1, E1c: training with and without the survival term | `rq1_silence.tex` | 8 (seeds 1–2) | 12 |
| RQ3: six inputs, two budgets, NAMED-OUT, less training data, FULL+GEN on Amazon reviews (2023 release, Cell Phones & Accessories) | `rq3_conditions.tex`, `rq3_strata.tex`, `rq3_inputs.tex`, `rq3_named.tex`, `rq3_gen.tex` | 20 (seed 1) | 60 |

Remaining runs can be added with `python scripts/experiments.py rq1 --gpus 0` and
`python scripts/experiments.py rq3 --gpus 0 --seeds 2 3`; finished runs are skipped.

## Notes for reading the tables

* **NLL** is in nats per event and includes the in-bin term, so it is a continuous
  negative log-likelihood that can be compared with other temporal point processes.
* **E1b:** the NLL in `rq1_bench.tex` is the density at the recorded waiting times. On data
  with rounded or repeated timestamps (Retweet, Taobao) this can favour whichever model makes
  the sharpest peaks. `scripts/rescore.py` scores every model by the probability of the
  recorded interval `[tau, tau + delta)` instead; those scores will be added here when the
  rescoring has been run.
* **RQ3:** one seed. The format-only control (SCRAMBLED vs. NO-TEXT, which should be 0) differs
  by about 0.03 nats, so smaller differences are within training noise. The ± values come from
  a paired bootstrap over users and do not include training randomness.

## Rebuilding the tables

```bash
python scripts/make_tables.py --runs runs --out paper_tables
```

The RQ1 tables need only the `summary.json` files (e.g. `--runs results/run_summaries`). The
RQ3 tables also need the per-event files of the runs and `data/amazon_phones`.
