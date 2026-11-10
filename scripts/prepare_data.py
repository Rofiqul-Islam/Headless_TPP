#!/usr/bin/env python
"""Prepare datasets in the common format (see htpp/data.py).

  # synthetic multivariate Hawkes with a known likelihood (RQ1)
  python scripts/prepare_data.py hawkes --out data/hawkes

  # EasyTPP benchmarks from the Hugging Face hub (or --local_dir with EasyTPP files)
  python scripts/prepare_data.py easytpp --name retweet --out data/retweet
  python scripts/prepare_data.py easytpp --name taobao --out data/taobao_cut --censor random_cut

  # Amazon Reviews 2023 with review text (RQ3): download, then build
  python scripts/prepare_data.py amazon-download --category Cell_Phones_and_Accessories --out raw/amazon
  python scripts/prepare_data.py amazon --reviews raw/amazon/Cell_Phones_and_Accessories.jsonl.gz \
      --meta raw/amazon/meta_Cell_Phones_and_Accessories.jsonl.gz --out data/amazon_phones
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from htpp.data import save_dataset  # noqa: E402
from htpp import sources  # noqa: E402

AMAZON_HF = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"


def amazon_download(category: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    wanted = [(f"review_categories/{category}.jsonl", out / f"{category}.jsonl"),
              (f"meta_categories/meta_{category}.jsonl", out / f"meta_{category}.jsonl")]
    for rel, dest in wanted:
        if dest.exists() or Path(str(dest) + ".gz").exists():
            print(f"exists: {dest}")
            continue
        ok = False
        try:   # 1) Hugging Face hub
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(AMAZON_HF, f"raw/{rel}", repo_type="dataset")
            shutil.copy(p, dest)
            ok = True
        except Exception as e:  # noqa: BLE001
            print(f"hub download failed for {rel}: {e}")
        if not ok:  # 2) the dataset's own web server (gzip)
            url = f"{AMAZON_URL}/{rel}.gz"
            try:
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, str(dest) + ".gz")
                ok = True
            except Exception as e:  # noqa: BLE001
                print(f"web download failed for {url}: {e}")
        if not ok:
            raise SystemExit(f"Could not download {rel}. Download it manually from https://amazon-reviews-2023.github.io "
                             f"and place it at {dest} (or {dest}.gz).")
        print(f"saved {dest}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hawkes")
    h.add_argument("--out", required=True)
    h.add_argument("--K", type=int, default=3)
    h.add_argument("--T", type=float, default=60.0)
    h.add_argument("--n_train", type=int, default=2000)
    h.add_argument("--n_val", type=int, default=300)
    h.add_argument("--n_test", type=int, default=500)
    h.add_argument("--branching_ratio", type=float, default=0.6)
    h.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("easytpp")
    e.add_argument("--name", required=True, help="retweet | stackoverflow | taobao | amazon | taxi")
    e.add_argument("--out", required=True)
    e.add_argument("--local_dir", default=None)
    e.add_argument("--censor", default="none", choices=["none", "random_cut"])
    e.add_argument("--split_long", type=int, default=None)
    e.add_argument("--resolution", type=float, default=None)
    e.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("amazon-download")
    d.add_argument("--category", required=True)
    d.add_argument("--out", default="raw/amazon")
    a = sub.add_parser("amazon")
    a.add_argument("--reviews", nargs="+", required=True)
    a.add_argument("--meta", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--cat_level", type=int, default=2)
    a.add_argument("--K", type=int, default=24)
    a.add_argument("--min_events", type=int, default=5)
    a.add_argument("--max_events_per_user", type=int, default=200)
    a.add_argument("--max_users", type=int, default=20000)
    a.add_argument("--desc_max_chars", type=int, default=400)
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "hawkes":
        splits, meta = sources.build_hawkes(args.n_train, args.n_val, args.n_test, args.K, args.T, args.seed,
                                            args.branching_ratio)
    elif args.cmd == "easytpp":
        splits, meta = sources.build_easytpp(args.name, args.local_dir, args.censor, args.split_long, args.seed,
                                             args.resolution)
    elif args.cmd == "amazon-download":
        amazon_download(args.category, Path(args.out))
        return
    else:
        splits, meta = sources.build_amazon(args.reviews, args.meta, args.cat_level, args.K, args.min_events,
                                            args.max_events_per_user, args.max_users, args.desc_max_chars, args.seed)
    save_dataset(args.out, splits, meta)
    print(f"saved {args.out}: " + ", ".join(f"{k} {len(v)} seqs / {sum(len(s) for s in v)} events" for k, v in splits.items()))


if __name__ == "__main__":
    main()
