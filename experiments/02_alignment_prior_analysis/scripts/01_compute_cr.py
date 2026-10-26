#!/usr/bin/env python3
"""Phase 1 (Experiment 02): stream corpora from Hugging Face, compute LZ4 compression ratio (CR).

Protocol: README.md Phase 1. CR(d) = len(lz4.frame.compress(raw)) / len(raw), default LZ4 frame level.
Documents with < 512 raw UTF-8 bytes are dropped before counting toward N.

Sampling: HF streaming with `IterableDataset.shuffle(seed=42, buffer_size=BUF)` (shuffles shard order and
uses a reservoir-style buffer), then keeps the first N docs that pass the length filter.

Outputs:
  results/cr_pilot.csv                  (doc_id, corpus, cr, byte_length, domain)
  data_cache/<corpus>_texts.parquet     (doc_id, text, domain) -- local cache, git-ignored

Usage: python 01_compute_cr.py --corpus fineweb_edu --n 20000
"""
import argparse
import json
import os
import sys
import time

import lz4.frame
import pandas as pd

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(EXP_DIR, "results")
CACHE = os.path.join(EXP_DIR, "data_cache")

CORPORA = {
    "fineweb_edu": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train"),
    "pile": dict(path="monology/pile-uncopyrighted", name=None, split="train"),
    "dclm": dict(path="mlfoundations/dclm-baseline-1.0", name=None, split="train"),
}
MIN_BYTES = 512


def compression_ratio(raw: bytes) -> float:
    return len(lz4.frame.compress(raw)) / len(raw)


def domain_of(ex: dict, corpus: str) -> str:
    if corpus == "pile":
        meta = ex.get("meta") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return meta.get("pile_set_name", "unknown")
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer", type=int, default=10000)
    ap.add_argument("--max-seen", type=int, default=200000, help="safety cap on streamed docs")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    cfg = CORPORA[args.corpus]
    t0 = time.time()
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer)

    rows, texts = [], []
    seen = short = 0
    t_cr = 0.0
    for ex in ds:
        seen += 1
        text = ex.get("text") or ""
        raw = text.encode("utf-8")
        if len(raw) < MIN_BYTES:
            short += 1
        else:
            t1 = time.perf_counter()
            cr = compression_ratio(raw)
            t_cr += time.perf_counter() - t1
            doc_id = f"{args.corpus}_{len(rows):06d}"
            dom = domain_of(ex, args.corpus)
            rows.append((doc_id, args.corpus, cr, len(raw), dom))
            texts.append((doc_id, text, dom))
            if len(rows) % 2000 == 0:
                print(f"[{args.corpus}] kept={len(rows)} seen={seen} elapsed={time.time()-t0:.0f}s", flush=True)
        if len(rows) >= args.n or seen >= args.max_seen:
            break

    df = pd.DataFrame(rows, columns=["doc_id", "corpus", "cr", "byte_length", "domain"])
    out_csv = os.path.join(RESULTS, f"cr_pilot_{args.corpus}.csv")
    df.to_csv(out_csv, index=False)
    pd.DataFrame(texts, columns=["doc_id", "text", "domain"]).to_parquet(
        os.path.join(CACHE, f"{args.corpus}_texts.parquet"), index=False)
    meta = dict(corpus=args.corpus, hf=cfg, n_kept=len(rows), n_seen=seen, n_short_dropped=short,
                min_bytes=MIN_BYTES, seed=args.seed, shuffle_buffer=args.buffer,
                sampling="HF streaming .shuffle(seed, buffer_size) then first N docs passing the >=512-byte filter",
                wall_seconds=time.time() - t0, lz4_cr_seconds=t_cr,
                lz4_docs_per_sec=len(rows) / t_cr if t_cr else None)
    with open(os.path.join(RESULTS, f"cr_pilot_{args.corpus}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
