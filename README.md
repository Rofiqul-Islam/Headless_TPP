# Headless-TPP

Code for the depth-exam response *LLMs as Temporal Point Processes: Token Likelihoods and Event Text*.
It implements, with a real fine-tuned LLM, everything the paper describes:

| Paper | Code |
|---|---|
| RQ1, Ideas 1–2: hierarchical time tokens, exact density and survival from token probabilities (Eqs. 4–9) | `htpp/timecode.py` |
| R1/R2: one spelling per bin, constrained softmax (used in training **and** evaluation) | `htpp/layout.py`, `htpp/model.py` |
| No separate head: new tokens get an input row and an output row of the LLM itself | `htpp/model.py` |
| Idea 3: survival term for the final quiet gap | `htpp/model.py` (`logS_gap`) |
| Idea 4: four training stages (grammar, next event, silence; calibration after training) | `htpp/train.py`, `htpp/evaluate.py` |
| E1a / E1b / E1c | `scripts/experiments.py rq1` |
| RQ3: six inputs (NO-TEXT, NAMES, SCRAMBLED, FULL, SHUFFLED, TEXT-ONLY), event and token budgets, NAMED-OUT | `htpp/layout.py`, `scripts/experiments.py rq3` |
| Section 4.3: writing the next description (FULL+GEN, loss with λ_d) | `htpp/model.py` (`desc_logprobs`), `htpp/generate.py` |
| Amazon protocol: chronological split, unseen users, strata, user-level bootstrap | `htpp/sources.py`, `scripts/make_tables.py` |
| Neural TPP baselines with an exact likelihood (GRU / Transformer + log-normal mixture) | `htpp/baselines.py` |

Every number in the result tables is produced by these scripts; nothing is filled in by hand.

---

## 1. Setup on the server

```bash
conda create -n htpp python=3.11 -y && conda activate htpp
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA version
pip install -r requirements.txt
huggingface-cli login        # optional; Qwen2.5 is not gated, but logging in avoids rate limits
```

Check that everything works (CPU is enough, takes about a minute):

```bash
python tests/make_fixtures.py          # tiny random model + tiny datasets, no downloads
python -m pytest -q tests/test_core.py # 8 tests: densities integrate to 1, survival = 1 - CDF,
                                       # every event scored once, masks, tree expansion, losses
python scripts/run.py --config configs/tiny.yaml \
  --set model.name=tests/tiny_model data.dir=tests/data/text_tiny data.condition=full out_dir=runs/smoke
```

The tiny model is random, so its numbers mean nothing; it only checks the pipeline.

## 2. Prepare the data

```bash
bash scripts/prepare_all.sh
```

**Windows PowerShell:** From the project root, activate the `htpp` Conda environment and run:

```powershell
.\scripts\prepare_all.ps1
```

If PowerShell blocks local scripts, use `powershell -ExecutionPolicy Bypass -File .\scripts\prepare_all.ps1` (process-only policy override). To select another Amazon category, pass `-AmazonCategory Category_Name`.


This creates, under `data/`:

| Folder | Used in | Source |
|---|---|---|
| `hawkes` | E1a | simulated 3-type Hawkes process (true likelihood stored per sequence) |
| `hawkes_short` | E1c | the same kind of process in short windows (many final quiet gaps) |
| `retweet`, `stackoverflow`, `taobao`, `amazon` | E1b | EasyTPP benchmarks from the Hugging Face hub (`easytpp/<name>`) |
| `taobao_cut` | E1c | Taobao cut into 32-event pieces that end at a random time |
| `amazon_phones` | RQ3 | Amazon Reviews 2023, *Cell Phones & Accessories* (set `AMAZON_CATEGORY` to change) |

Notes:
* If the server cannot reach the Hugging Face hub, download the EasyTPP files yourself and use
  `python scripts/prepare_data.py easytpp --name taobao --local_dir <folder with train/dev/test .pkl or .json> --out data/taobao`.
* Amazon: the downloader tries the Hugging Face copy and then the dataset's web server. If both fail,
  download `<Category>.jsonl.gz` and `meta_<Category>.jsonl.gz` from https://amazon-reviews-2023.github.io
  into `raw/amazon/` and rerun. The 2018 release (Ni et al., the version used by LAMP and Language-TPP)
  also works: pass its review and meta files to `prepare_data.py amazon`.
* The Amazon builder uses the chronological protocol of Section 4: 10% of users are held out entirely
  (`test_unseen`); for the rest, events before the 80% time quantile are training, up to the 90%
  quantile validation, and later events test. Earlier events of a user stay in the history as context
  (`n_hist`) but are scored only once. Event types are the 23 most frequent categories at depth 2 of the
  category path plus "Other".

## 3. Run the experiments

```bash
python scripts/experiments.py rq1 --gpus 0,1,2,3      # E1a, E1b, E1c: 90 runs
python scripts/experiments.py rq3 --gpus 0,1,2,3      # RQ3: 60 runs
```

* Each job is one `scripts/run.py` call with its own folder in `runs/`. Finished jobs are skipped, so you
  can stop and restart at any time. `--dry_run` prints the commands.
* Defaults: Qwen2.5-1.5B with LoRA (rank 16), three seeds. Change with `--model Qwen/Qwen2.5-7B` or
  `--seeds 1 2 3 4 5`. The no-pre-training ablation uses the Qwen2.5-0.5B architecture (`--small_model`).
* Small jobs (baselines, 0.5B) can share a GPU: `--jobs_per_gpu 2`.
* Rough cost on one A100 80GB with 1.5B: a benchmark run takes from tens of minutes (StackOverflow,
  Taobao) to a few hours (Retweet, about 2M events); an RQ3 text run takes about 1–2 hours. Run one job
  first and check `runs/<job>/log.txt` before launching everything.

What is run:

| Experiment | Runs per seed | Folder |
|---|---|---|
| E1a: 8x8x8, 32x16, one-level (512) tokens, digit strings, GRU and Transformer baselines | 6 | `runs/rq1_hawkes/` |
| E1b: Headless-TPP, digits, no pre-training, GRU, Transformer on 4 datasets | 20 | `runs/rq1_bench/` |
| E1c: with and without the survival term on 2 datasets | 4 | `runs/rq1_silence/` |
| RQ3: 6 inputs x 2 budgets | 12 | `runs/rq3/{event,token}/` |
| RQ3: NAMED-OUT for NO-TEXT and FULL | 2 | `runs/rq3/named_out/` |
| RQ3: FULL and NO-TEXT with 10% and 30% of training users | 4 | `runs/rq3/train_frac*/` |
| RQ3: FULL+GEN with λ_d = 0.1 and 1 | 2 | `runs/rq3/gen/` |

"Without calibration" (stage 4) is read from the same runs (`metrics_raw`), so it needs no extra job.

## 4. Make the tables for the paper

```bash
python scripts/make_tables.py --runs runs --out paper_tables
```

This writes `rq1_hawkes.tex`, `rq1_bench.tex`, `rq1_silence.tex`, `rq3_conditions.tex`, `rq3_strata.tex`,
`rq3_named.tex`, `rq3_gen.tex`, the figure `rq3_history.pdf`, and `results.json` (all numbers).
Upload the `.tex` files to a folder named **`generated/`** in the Overleaf project. The paper includes
each table automatically when its file exists (`\resulttable{...}` in `main.tex`), and the sentences
that say "results will be added" switch to point at the tables. Then write the discussion from
`results.json`.

## 5. One run by hand

```bash
python scripts/run.py --config configs/rq3.yaml \
  --set data.condition=full data.budget=event train.seed=1 out_dir=runs/try/full
```

Outputs in `out_dir`: `log.txt`, `config.json`, `best.pt` (LoRA weights and new-token rows),
`summary.json` (metrics after calibration, `metrics_raw` before, `metrics_unseen` for unseen users,
temperatures, training history), `test_events.csv.gz` (one row per scored event: time and type NLL,
predicted type, PIT value, predicted mean/median waiting time, text features), `test_gaps.csv.gz`
(silence NLL per sequence), and `generations.jsonl` for FULL+GEN.

Any config value can be overridden with `--set key.sub=value`, e.g. `model.name=Qwen/Qwen2.5-7B`,
`timecode.branching=[16,8,8]`, `data.max_events=64`, `train.lambda_d=0.1 train.gen=true`.

## 6. What the code computes, exactly

* **Event layout.** `[c1 c2 c3] [type] [name words]? [<desc> review </desc>]? [</e>]`. Probabilities are
  read at the position before each target token, with a softmax restricted to the tokens the grammar
  allows there (8 or 9 time tokens, K type tokens, or ordinary words plus `</desc>`).
* **NLL.** time part `-sum log q_l - log u_B` (includes the in-bin term, so it is a continuous NLL in
  nats per event), type part `-log P(k | bin, history)`, silence part `-log S(g)` for each sequence's
  final quiet gap, all divided by the number of scored events. The three parts add up exactly (Eq. 11).
* **Windows.** Every event except the first of a sequence (or the history part, `n_hist`) is scored
  exactly once. Event budget: 32 (RQ3) or 128 (RQ1) events per window, half of them context.
  Token budget: 1,024 tokens per window, up to half used by earlier events as context.
* **TEXT-ONLY.** The type token must stay in the sequence because it is a target, so a custom attention
  mask hides it from every later position (tested in `tests/test_core.py`).
* **Time prediction.** The mean and median waiting time are computed exactly from the token tree: for
  each predicted event, every branch of the first two levels is appended with a tree attention mask, so
  one forward pass gives all 1 + 8 + 64 level distributions (tested against a naive forward pass).
* **Type accuracy** is the most probable type given the history and the true time bin. The baselines'
  type distribution does not depend on time, so for them it is the same as the marginal prediction.
* **Calibration (stage 4).** One temperature per time level and one for the type token, fitted on
  validation NLL and used for density, survival and intensity alike. `metrics_raw` is the result without it.
* **Strata (RQ3).** Paired per-event gains of FULL over NO-TEXT, averaged over seeds; intervals come from a
  bootstrap over users. Consecutive-review similarity uses the mean input embedding of the LLM.

## 7. Practical notes

* `model.attn_implementation=sdpa` is required for TEXT-ONLY and for time prediction (custom masks).
  `flash_attention_2` works for the other conditions if you set `eval.predict_max_events=0`.
* Out of memory: lower `train.max_tokens_per_batch` (and raise `train.grad_accum`), or
  `train.eval_tokens_per_batch`. Gradient checkpointing is on by default.
* The 7B model with LoRA fits in 80 GB with the defaults; for 40 GB use `train.max_tokens_per_batch=8192`.
* Leakage: Amazon reviews before the model's training cut-off may be memorised. `generations.jsonl` and
  `summary.json` report the share of generated 8-grams found in the training reviews.
