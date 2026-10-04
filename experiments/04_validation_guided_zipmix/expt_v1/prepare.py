#!/usr/bin/env python3
"""Freeze leakage-separated, token-block data for the Zip-Mix mechanism screen.

No model calls. The inherited public Pile pilot is a convenience sample, not a
new random sample from the entire Pile. Raw text and token arrays stay ignored.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

import lz4.frame
import numpy as np
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

BOUNDARIES = np.array([.50, .60, .67, .73, .80, .90])
TARGET = {"PubMed Abstracts", "PubMed Central"}
_TARGET_BYTES = []
_TARGET_COMPRESSED = []


def digest(value):
    return hashlib.sha256(value).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def grams(text, n=13):
    words = re.findall(r"\w+", text.casefold())
    return {digest(" ".join(words[i:i+n]).encode())[:24]
            for i in range(len(words)-n+1)}


def init_scoring(targets):
    global _TARGET_BYTES, _TARGET_COMPRESSED
    _TARGET_BYTES = targets
    _TARGET_COMPRESSED = [len(gzip.compress(v, compresslevel=6, mtime=0)) for v in targets]


def alignment(raw):
    """Equal-byte windows limit document-length confounding; preserve negatives."""
    x = raw[:1024]
    cx = len(gzip.compress(x, compresslevel=6, mtime=0))
    scores = [1 - (len(gzip.compress(x+y, compresslevel=6, mtime=0))-min(cx, cy))/max(cx, cy)
              for y, cy in zip(_TARGET_BYTES, _TARGET_COMPRESSED)]
    return max(scores), float(np.mean(scores))


def split_rows(frame):
    """Identity-based ordering within domain: no model/score-dependent split."""
    rows, seen, duplicates = [], set(), 0
    counts = Counter(frame.domain)
    # Sparse sources cannot provide a stable per-domain control at this scale.
    domains = sorted(d for d, n in counts.items() if n >= 250)
    for row in frame.itertuples():
        if row.domain not in domains:
            continue
        identity = digest(normalized(row.text).encode())
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        raw = row.text.encode()[:8192]
        if len(raw) < 1024:
            continue
        text = raw.decode("utf-8", errors="ignore")
        rows.append(dict(id=identity, domain=row.domain, text=text,
                         order=digest(("zipmix-04-split-2026:"+identity).encode())))
    train, dev, test = [], [], []
    for domain in domains:
        subset = sorted((r for r in rows if r['domain']==domain), key=lambda r:r['order'])
        if len(subset)<100:
            continue
        dev += subset[:32]
        test += subset[32:96]
        train += subset[96:]
    return train, dev, test, duplicates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--vocab-size', type=int, default=8192)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if (args.output/'manifest.json').exists():
        raise SystemExit('Frozen data exists; use a new directory for changed preparation.')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame = pd.read_parquet(args.source)
    train, dev, test, duplicates = split_rows(frame)
    # Decontaminate using text only, never loss/accuracy. Apply to all candidate
    # data before any method sees it; record every removal by document hash.
    held_grams = set().union(*(grams(r['text']) for r in dev+test))
    clean, removed = [], []
    for r in train:
        (removed if grams(r['text']) & held_grams else clean).append(r)
    train = clean
    domains = sorted({r['domain'] for r in train})
    domain_id = {d:i for i,d in enumerate(domains)}
    if not TARGET.issubset(domain_id):
        raise ValueError('Target domains missing after decontamination')
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator((r['text'] for r in train),
        trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=['<unk>', '<eos>']))
    tokenizer.save(str(args.output/'tokenizer.json'))
    targets = [r['text'].encode()[:1024] for r in dev if r['domain'] in TARGET][:64]
    with ProcessPoolExecutor(args.workers, initializer=init_scoring, initargs=(targets,)) as pool:
        scores = list(pool.map(alignment, (r['text'].encode() for r in train), chunksize=32))
    block_rows = []
    for r, (score, mean_score) in zip(train, scores):
        raw = r['text'].encode()
        cr = len(lz4.frame.compress(raw))/len(raw)
        ids = tokenizer.encode(r['text']).ids + [tokenizer.token_to_id('<eos>')]
        # Fixed blocks are the sampling unit; no padding or cross-document
        # packing. The remainder is excluded identically for every method.
        for start in range(0, min(len(ids)-args.seq_len, 2048), args.seq_len):
            block_rows.append((ids[start:start+args.seq_len+1], domain_id[r['domain']],
                               int(np.searchsorted(BOUNDARIES, cr, side='right')),
                               score, mean_score, cr, r['id']))
    if not block_rows:
        raise ValueError('No training blocks')
    np.savez_compressed(args.output/'train.npz',
        tokens=np.array([r[0] for r in block_rows], dtype=np.int32),
        domain=np.array([r[1] for r in block_rows], dtype=np.int32),
        bucket=np.array([r[2] for r in block_rows], dtype=np.int32),
        score=np.array([r[3] for r in block_rows], dtype=np.float32),
        mean_score=np.array([r[4] for r in block_rows], dtype=np.float32),
        cr=np.array([r[5] for r in block_rows], dtype=np.float32),
        doc_id=np.array([r[6] for r in block_rows]))

    def eval_rows(rows):
        out = []
        for r in rows:
            if r['domain'] not in domain_id:
                continue
            ids = tokenizer.encode(r['text']).ids
            if len(ids)>args.seq_len:
                out.append((ids[:args.seq_len+1], domain_id[r['domain']], r['id']))
        return out

    dev_rows = eval_rows([r for r in dev if r['domain'] in TARGET])
    target_rows = eval_rows([r for r in test if r['domain'] in TARGET])
    broad_rows = eval_rows([r for r in test if r['domain'] not in TARGET])
    np.savez_compressed(args.output/'dev.npz',
        tokens=np.array([r[0] for r in dev_rows], dtype=np.int32),
        domain=np.array([r[1] for r in dev_rows], dtype=np.int32))
    np.savez_compressed(args.output/'eval.npz',
        target_tokens=np.array([r[0] for r in target_rows], dtype=np.int32),
        target_domain=np.array([r[1] for r in target_rows], dtype=np.int32),
        broad_tokens=np.array([r[0] for r in broad_rows], dtype=np.int32),
        broad_domain=np.array([r[1] for r in broad_rows], dtype=np.int32))
    split_manifest = {name:[{'id':r['id'],'domain':r['domain']} for r in rows]
                      for name,rows in [('train',train),('dev',dev),('test',test),('removed_overlap',removed)]}
    (args.output/'split_manifest.json').write_text(json.dumps(split_manifest, indent=2)+'\n')
    buckets = np.array([r[2] for r in block_rows])
    clipped = np.maximum(np.array([r[3] for r in block_rows]), 0)
    base = np.bincount(buckets,minlength=7).astype(float);base/=base.sum()
    prior = np.bincount(buckets, weights=clipped,minlength=7)
    prior[base>0] += 1e-3
    prior/=prior.sum()
    metadata = dict(source_sha256=digest(args.source.read_bytes()),
        source='Experiment 02 cached monology/pile-uncopyrighted streaming pilot; 20000 source documents',
        source_hub_revision='unavailable in inherited pilot; frozen local source hash recorded',
        target_domains=sorted(TARGET), domains=domain_id,
        train_documents=len(train), duplicate_documents=duplicates,
        decontamination_removed_documents=len(removed), train_blocks=len(block_rows),
        target_eval_blocks=len(target_rows), broad_eval_blocks=len(broad_rows), dev_blocks=len(dev_rows),
        vocab_size=tokenizer.get_vocab_size(), seq_len=args.seq_len,
        train_token_capacity=len(block_rows)*args.seq_len,
        score_definition='original max(1-NCD); mean_score is published ZIP-FIT aggregation; gzip level6 fixed1024-byte prefix windows',
        score_scale='raw similarities can be negative; sampling clips to zero explicitly',
        raw_negative_max_scores=int(sum(s[0]<0 for s in scores)),
        raw_negative_mean_scores=int(sum(s[1]<0 for s in scores)),
        compression='lz4.frame default on first8192-byte document prefix; upper bucket open-ended',
        sampling_unit='128-token within-document block; alpha sums block scores; token-proportional extension of document formula',
        decontamination='normalized exact full-document dedup plus any shared13-word-gram in retained8192-byte prefixes; train vs dev/test',
        baseline_bucket_mass=base.tolist(), zipmix_bucket_mass=prior.tolist(),
        total_variation_from_token_proportional=float(.5*np.abs(prior-base).sum()),
        seconds=time.time()-started,
        files={f.name:digest(f.read_bytes()) for f in args.output.iterdir() if f.is_file()})
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
