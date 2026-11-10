"""Create a tiny random Qwen2 model with its own small tokenizer, and tiny datasets,
so the whole pipeline can be tested on a CPU without downloading anything."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from htpp import sources  # noqa: E402
from htpp.data import EventSeq, save_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
WORDS = ("great sound charger died after a week battery case screen cable fast slow cheap broke works fine love it "
         "returned replacement product quality price perfect fits phone good bad not recommend would buy again "
         "headphones earbuds mount stand holder adapter wireless bluetooth").split()


def make_tiny_model(out: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    rng = np.random.default_rng(0)
    corpus = [" ".join(rng.choice(WORDS, 12)) + f" {rng.integers(0, 1000)}" for _ in range(2000)]
    corpus += ["0 1 2 3 4 5 6 7 8 9", "The following is a sequence of events. Each event gives its waiting time, "
               "its type and optional text.", "type Electronics Phones Cases Chargers Other"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, tr)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)
    cfg = Qwen2Config(vocab_size=len(tok) + 8, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    print("tiny model:", out, "vocab", len(tok), "rows", cfg.vocab_size)


def make_text_dataset(out: Path, n_users=60, K=5, seed=0):
    rng = np.random.default_rng(seed)
    names = ["Phones", "Cases", "Chargers", "Cables", "Other"][:K]
    seqs = []
    for u in range(n_users):
        n = int(rng.integers(4, 14))
        tau = rng.exponential(10.0, n)
        k = rng.integers(0, K, n)
        # replacement reviews ("broke") make the next purchase come sooner and in the same type
        text = []
        for i in range(n):
            broke = rng.random() < 0.3
            text.append(("broke after a week " if broke else "works fine ") + " ".join(rng.choice(WORDS, 5)))
            if broke and i + 1 < n:
                tau[i + 1] *= 0.2
                k[i + 1] = k[i]
        t = np.cumsum(tau)
        seqs.append(EventSeq(f"u{u}", t, k, text=text, shuf=list(rng.permutation(text)), t_end=float(t[-1] + 20)))
    idx = rng.permutation(n_users)
    splits = {"train": [seqs[i] for i in idx[:40]], "val": [seqs[i] for i in idx[40:50]],
              "test": [seqs[i] for i in idx[50:]]}
    meta = {"source": "tiny-text", "K": K, "type_names": names, "scrambled_names": ["zuvo", "kepa", "rido", "mavu", "tobe"][:K],
            "type_parent": [0, 0, 1, 1, 2][:K], "time_unit": "days", "time_resolution": 1e-4, "has_text": True}
    save_dataset(out, splits, meta)


def make_fake_amazon_raw(out: Path, n_users=120, seed=0, old_format=False):
    """Raw review/meta files in the Amazon Reviews 2023 (or 2018) format, for testing the builder."""
    import gzip
    import json
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    cats = [["Cell Phones", "Accessories", "Chargers"], ["Cell Phones", "Accessories", "Cases"],
            ["Cell Phones", "Accessories", "Cables"], ["Cell Phones", "Phones", "Unlocked"],
            ["Cell Phones", "Accessories", "Screen Protectors"], ["Cell Phones", "Phones", "Carrier"]]
    items = [f"B{i:05d}" for i in range(200)]
    with gzip.open(out / "meta.jsonl.gz", "wt") as f:
        for it in items:
            c = cats[int(rng.integers(len(cats)))]
            row = {"asin": it, "category": c} if old_format else {"parent_asin": it, "categories": c, "title": "x"}
            f.write(json.dumps(row) + "\n")
    with gzip.open(out / "reviews.jsonl.gz", "wt") as f:
        for u in range(n_users):
            t = 1.4e9 + rng.uniform(0, 2e8)
            for _ in range(int(rng.integers(3, 25))):
                t += rng.exponential(3e6)
                it = items[int(rng.integers(len(items)))]
                txt = " ".join(rng.choice(WORDS, 8))
                if old_format:
                    row = {"reviewerID": f"U{u}", "asin": it, "unixReviewTime": int(t), "summary": "ok", "reviewText": txt}
                else:
                    row = {"user_id": f"U{u}", "parent_asin": it, "timestamp": int(t * 1000), "title": "ok", "text": txt}
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    make_tiny_model(HERE / "tiny_model")
    splits, meta = sources.build_hawkes(n_train=60, n_val=15, n_test=20, K=3, T=15.0, seed=0)
    save_dataset(HERE / "data" / "hawkes_tiny", splits, meta)
    make_text_dataset(HERE / "data" / "text_tiny")
    for old in (False, True):
        raw = HERE / "data" / ("amazon_raw_2018" if old else "amazon_raw_2023")
        make_fake_amazon_raw(raw, old_format=old)
        splits, meta = sources.build_amazon([raw / "reviews.jsonl.gz"], [raw / "meta.jsonl.gz"], cat_level=2, K=5,
                                            min_events=4, max_users=None)
        save_dataset(HERE / "data" / ("amazon_tiny_2018" if old else "amazon_tiny"), splits, meta)
        print("amazon", "2018" if old else "2023", {k: len(v) for k, v in splits.items()}, meta["type_names"])
    print("fixtures ready")
