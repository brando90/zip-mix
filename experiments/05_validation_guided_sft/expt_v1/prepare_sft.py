#!/usr/bin/env python3
"""Download pinned public data/tokenizer; freeze the six-method SFT screen."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import platform
import time

import datasets
from huggingface_hub import HfApi
import lz4.frame
import numpy as np
from transformers import AutoTokenizer

from common import (CONFIG, LABELS, METHODS, MODEL, MODEL_REVISION, SEEDS, SOURCES,
                    atomic_json, canonical_record, content_grams, file_hash,
                    mixture_weights, now, sha256, shuffled_bucket_weights, stable_hash)

_VIEWS, _VIEW_C = [], []


def byte_view(text):
    return text.encode("utf-8")[:CONFIG["score_byte_window"]].decode("utf-8", errors="ignore").encode("utf-8")


def initialize_scores(views):
    global _VIEWS, _VIEW_C
    _VIEWS = views
    _VIEW_C = [len(gzip.compress(v, compresslevel=CONFIG["gzip_level"], mtime=0)) for v in views]


def compression_score(content):
    raw = content.encode("utf-8")
    view = byte_view(content)
    cx = len(gzip.compress(view, compresslevel=CONFIG["gzip_level"], mtime=0))
    alignment = [1 - (len(gzip.compress(view + v, compresslevel=CONFIG["gzip_level"], mtime=0)) - min(cx, cv)) / max(cx, cv)
                 for v, cv in zip(_VIEWS, _VIEW_C)]
    if not alignment:
        raise ValueError("Empty alignment development set")
    return (len(lz4.frame.compress(raw)) / len(raw), float(np.max(alignment)), float(np.mean(alignment)), len(view))


def convert(rows, domain, split):
    out, rejected = [], Counter()
    for row in rows:
        try:
            out.append(canonical_record(row, domain, split))
        except (ValueError, KeyError) as exc:
            rejected[str(exc)] += 1
    return sorted(out, key=lambda r: r["id"]), dict(rejected)


def encode_record(row, tokenizer, label_tokens):
    tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
    # Assert prompt/continuation boundary as well as isolated-token length.
    for i in range(len(row["choices"])):
        full = tokenizer.encode(row["prompt"] + " " + LABELS[i], add_special_tokens=False)
        if full != tokens + [label_tokens[i]]:
            raise ValueError("Prompt/answer tokenization boundary changed")
    if len(tokens) + 1 > CONFIG["max_input_length"]:
        return None
    return {**row, "tokens": tokens, "answer_token": label_tokens[row["answer"]]}


def save_arrays(path, records, pad_token):
    n, width = len(records), CONFIG["max_input_length"] - 1
    matrix = np.full((n, width), pad_token, dtype=np.int32)
    for i, row in enumerate(records):
        matrix[i, :len(row["tokens"])] = row["tokens"]
    np.savez_compressed(path, input_ids=matrix,
        lengths=np.array([len(r["tokens"]) for r in records], dtype=np.int32),
        answer=np.array([r["answer"] for r in records], dtype=np.int32),
        answer_token=np.array([r["answer_token"] for r in records], dtype=np.int32),
        nchoices=np.array([len(r["choices"]) for r in records], dtype=np.int32),
        ids=np.array([r["id"] for r in records]),
        question_ids=np.array([r["question_id"] for r in records]),
        domain=np.array([r["domain"] for r in records]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Preparation output must be new/empty; frozen data cannot be overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    api = HfApi()
    if api.model_info(MODEL, revision=MODEL_REVISION).sha != MODEL_REVISION:
        raise ValueError("Model revision failed immutable resolution")
    for repo, _, revision in SOURCES.values():
        if api.dataset_info(repo, revision=revision).sha != revision:
            raise ValueError("Dataset revision failed immutable resolution")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, cache_dir=args.cache_dir)
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    label_enc = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in LABELS]
    if any(len(x) != 1 for x in label_enc) or len({x[0] for x in label_enc}) != len(LABELS):
        raise ValueError("Answer labels must be five distinct single tokens")
    label_tokens = [x[0] for x in label_enc]
    tokenizer.save_pretrained(args.output / "tokenizer")
    loaded, rejected = {}, {}
    for domain, (repo, config, revision) in SOURCES.items():
        kwargs = dict(path=repo, revision=revision, cache_dir=str(args.cache_dir) if args.cache_dir else None)
        if config is not None:
            kwargs["name"] = config
        loaded[domain] = datasets.load_dataset(**kwargs)
    pools = {}
    for domain, dataset in loaded.items():
        pools[domain], rejected[domain + ":train"] = convert(dataset["train"], domain, "train")
    held, held_sources = {}, {"alignment": ("sciq", "validation"),
                             "target": ("sciq", "test"),
                             "retention": ("commonsense_qa", "validation")}
    for role, (domain, split) in held_sources.items():
        held[role], rejected[role] = convert(loaded[domain][split], domain, split)
    protected_ids = {r["id"] for rows in held.values() for r in rows}
    protected_questions = {r["question_id"] for rows in held.values() for r in rows}
    protected_grams = set().union(*(content_grams(r) for rows in held.values() for r in rows))
    train, removed, seen_questions = [], [], set()
    stats = {}
    for domain in sorted(pools):
        candidates = []
        counts = Counter()
        for record in pools[domain]:
            reason = None
            if record["id"] in protected_ids or record["question_id"] in protected_questions:
                reason = "heldout_exact"
            elif content_grams(record) & protected_grams:
                reason = "heldout_content_13gram"
            elif record["question_id"] in seen_questions:
                reason = "training_duplicate_question"
            if reason:
                removed.append({"id": record["id"], "domain": domain, "reason": reason})
                counts[reason] += 1
                continue
            encoded = encode_record(record, tokenizer, label_tokens)
            if encoded is None:
                counts["overlength"] += 1
                continue
            seen_questions.add(record["question_id"])
            candidates.append(encoded)
        chosen = candidates[:CONFIG["training_cap_per_domain"]]
        if not chosen:
            raise ValueError(f"Empty training domain after exclusions: {domain}")
        train.extend(chosen)
        stats[domain] = {"raw_valid_records": len(pools[domain]), "eligible": len(candidates),
                         "chosen": len(chosen), "exclusions": dict(counts)}
    eval_rows, seen_held_ids, seen_held_questions, seen_held_grams = {}, set(), set(), set()
    held_exclusions = {}
    # Priority is frozen: alignment, target, retention. Remove substantive
    # duplicates from later roles before their deterministic hash cap.
    for role in ("alignment", "target", "retention"):
        candidates, counts = [], Counter()
        role_questions = set()
        for row in held[role]:
            if row["question_id"] in role_questions:
                counts["within_role_duplicate_question"] += 1
                continue
            if row["id"] in seen_held_ids or row["question_id"] in seen_held_questions:
                counts["cross_role_exact"] += 1
                continue
            row_grams = content_grams(row)
            if row_grams & seen_held_grams:
                counts["cross_role_content_13gram"] += 1
                continue
            encoded = encode_record(row, tokenizer, label_tokens)
            if encoded is None:
                counts["overlength"] += 1
                continue
            role_questions.add(row["question_id"])
            candidates.append(encoded)
        cap = CONFIG["alignment_views"] if role == "alignment" else CONFIG["evaluation_cap_per_split"]
        chosen = candidates[:cap]
        if len(chosen) < cap:
            raise ValueError(f"Insufficient eligible {role} items: {len(chosen)} < {cap}")
        eval_rows[role] = chosen
        # Protect all eligible records from that role, not only the capped set.
        for row in candidates:
            seen_held_ids.add(row["id"])
            seen_held_questions.add(row["question_id"])
            seen_held_grams.update(content_grams(row))
        held_exclusions[role] = dict(counts)
    train = sorted(train, key=lambda r: (r["domain"], r["id"]))
    views = [byte_view(r["content"]) for r in eval_rows["alignment"]]
    if args.workers == 1:
        initialize_scores(views)
        scores = [compression_score(r["content"]) for r in train]
    else:
        with ProcessPoolExecutor(args.workers, initializer=initialize_scores, initargs=(views,)) as pool:
            scores = list(pool.map(compression_score, (r["content"] for r in train), chunksize=32))
    cr, alignment, mean_alignment, byte_lengths = map(np.asarray, zip(*scores))
    boundaries = np.quantile(cr, np.arange(1, CONFIG["compression_bins"]) / CONFIG["compression_bins"])
    bucket = np.searchsorted(boundaries, cr, side="right")
    domains = sorted(pools)
    domain_ids = np.array([domains.index(r["domain"]) for r in train])
    weights, diagnostics = mixture_weights(domain_ids, bucket, cr, alignment, [r["id"] for r in train], direct_scores=mean_alignment)
    weights["shuffled_zipmix"] = shuffled_bucket_weights(domain_ids, byte_lengths, bucket, alignment)
    for method in METHODS:
        value = weights[method]
        if not np.isfinite(value).all() or (value < 0).any() or not np.isclose(value.sum(), 1):
            raise ValueError(f"Invalid mixture: {method}")
    save_arrays(args.output / "train.npz", train, pad_token)
    for role in ("target", "retention"):
        save_arrays(args.output / f"{role}.npz", eval_rows[role], pad_token)
    exact_hits = sum(r["id"] in protected_ids or r["question_id"] in protected_questions for r in train)
    gram_hits = sum(bool(content_grams(r) & protected_grams) for r in train)
    if exact_hits or gram_hits:
        raise ValueError("Postfilter contamination invariant failed")
    np.savez_compressed(args.output / "selection.npz", cr=cr, score=alignment, mean_score=mean_alignment, bucket=bucket,
                        score_bytes=byte_lengths, domain=domain_ids,
                        **{method: weights[method] for method in METHODS})
    # Inspectable private provenance includes raw public source text; ignored
    # artifacts are never staged or published by this pipeline.
    with (args.output / "records.jsonl").open("w") as stream:
        for role, rows in [("train", train), *eval_rows.items()]:
            for row in rows:
                stream.write(json.dumps({"role": role, **row}, ensure_ascii=False) + "\n")
    atomic_json(args.output / "removed_records.json", removed)
    atomic_json(args.output / "identities.json", {
        "train": [r["id"] for r in train],
        **{role: [r["id"] for r in rows] for role, rows in eval_rows.items()}})
    relative_files = sorted(p.relative_to(args.output).as_posix() for p in args.output.rglob("*") if p.is_file())
    manifest = {
        "schema": 1, "created_at": now(), "model": MODEL, "model_revision": MODEL_REVISION,
        "source_revisions": {d: {"repository": r, "config": c, "revision": v} for d, (r, c, v) in SOURCES.items()},
        "config": CONFIG, "methods": list(METHODS), "seeds": list(SEEDS), "expected_training_cells": len(METHODS) * len(SEEDS),
        "label_tokens": label_tokens, "pad_token_id": pad_token, "domains": domains,
        "counts": {"train": len(train), **{r: len(x) for r, x in eval_rows.items()}},
        "domain_stats": stats, "conversion_rejections": rejected, "heldout_exclusions": held_exclusions,
        "protected_full_heldout_counts": {r: len(x) for r, x in held.items()},
        "leakage_rule": "Remove equal normalized questions/content and any shared 13-word contiguous content segment; scan question and choices separately, excluding wrappers. Protect full official heldout splits.",
        "postfilter_train_heldout_exact_hits": exact_hits, "postfilter_train_heldout_13gram_hits": gram_hits,
        "bucket_boundaries": boundaries.tolist(), "bucket_counts": np.bincount(bucket, minlength=CONFIG["compression_bins"]).tolist(),
        "score": {"zipmix_aggregation": "max (historical ZipMix variant, not published ZIP-FIT)", "direct_zipfit_aggregation": "mean (published ZIP-FIT)", "concatenation_order": "candidate_then_development", "clipping_for_bucket_weights": "max(score,0)",
                  "byte_length_min": int(byte_lengths.min()), "byte_length_max": int(byte_lengths.max()),
                  "alignment_min": float(alignment.min()), "alignment_max": float(alignment.max()), **diagnostics},
        "mixtures": {m: {"positive_examples": int((w > 0).sum()), "effective_examples": float(1 / np.square(w).sum()),
                         "domain_mass": {d: float(w[domain_ids == i].sum()) for i, d in enumerate(domains)},
                         "bucket_mass": np.bincount(bucket, weights=w, minlength=CONFIG["compression_bins"]).tolist()}
                     for m, w in weights.items()},
        "files": {name: file_hash(args.output / name) for name in relative_files},
        "preparation_code": {name: file_hash(Path(__file__).parent / name) for name in ("common.py", "prepare_sft.py")},
        "elapsed_seconds": time.monotonic() - start,
        "runtime": {"python": platform.python_version(), "datasets": datasets.__version__, "numpy": np.__version__},
    }
    manifest["manifest_id"] = stable_hash(manifest)
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"manifest_id": manifest["manifest_id"], "counts": manifest["counts"],
                      "elapsed_seconds": manifest["elapsed_seconds"], "mixtures": manifest["mixtures"]}, indent=2))


if __name__ == "__main__":
    main()
