#!/usr/bin/env bash
# Prepare every dataset used in the paper under data/. Needs internet access on the
# server (Hugging Face hub for EasyTPP; Hugging Face or UCSD for Amazon reviews).
set -euo pipefail
cd "$(dirname "$0")/.."

# RQ1 / E1a: synthetic Hawkes with a known likelihood (about 60 events per sequence)
python scripts/prepare_data.py hawkes --out data/hawkes --K 3 --T 60
# RQ1 / E1c: short observation windows, so many sequences end in a quiet gap
python scripts/prepare_data.py hawkes --out data/hawkes_short --K 3 --T 6 --n_train 6000 --n_val 800 --n_test 1500 --seed 1

# RQ1 / E1b: EasyTPP benchmarks (Hugging Face: easytpp/<name>)
for d in retweet stackoverflow taobao amazon; do
  python scripts/prepare_data.py easytpp --name "$d" --out "data/$d"
done
# RQ1 / E1c on real data: Taobao cut into short windows that end at a random time
python scripts/prepare_data.py easytpp --name taobao --out data/taobao_cut --censor random_cut --split_long 32

# RQ3: Amazon reviews with text (change the category if you like; see README)
CAT=${AMAZON_CATEGORY:-Cell_Phones_and_Accessories}
python scripts/prepare_data.py amazon-download --category "$CAT" --out raw/amazon
REV=$(ls raw/amazon/"$CAT".jsonl* | head -1)
META=$(ls raw/amazon/meta_"$CAT".jsonl* | head -1)
python scripts/prepare_data.py amazon --reviews "$REV" --meta "$META" --out data/amazon_phones \
  --cat_level 2 --K 24 --min_events 5 --max_users 20000

echo "all datasets ready in data/"
