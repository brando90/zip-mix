#!/usr/bin/env python3
"""Prospective fixed-byte pack calibration; preserves the untrained v1 data."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil
import time

import datasets
import lz4.frame
import numpy as np
from scipy.stats import spearmanr

from common import (CONFIG, METHODS, MODEL, MODEL_REVISION, SEEDS, SOURCES,
                    atomic_json, canonical_record, content_grams, file_hash,
                    now, sha256, stable_hash)


def make_packs(records, role, width=None):
    """Disjoint whole-record groups; raw UTF-8 prefix has exactly width bytes.

    A natural newline separates real documents. There is no padding, repetition,
    or score-dependent ordering. The final member may extend past the scoring
    prefix; all members inherit its pack score and train on their full prompts.
    Trailing groups with too few real bytes are explicitly dropped.
    """
    width = CONFIG["score_byte_window"] if width is None else width
    packs, pending, parts, size = [], [], [], 0
    for record in sorted(records, key=lambda r: r["id"]):
        raw = record["content"].encode("utf-8")
        if not raw:
            raise ValueError("Empty pack member")
        pending.append(record)
        parts.append(raw + b"\n")
        size += len(raw) + 1
        if size >= width:
            full = b"".join(parts)
            view = full[:width]
            assert len(view) == width
            pack_id = stable_hash({"role": role, "members": [r["id"] for r in pending], "width": width})
            packs.append({"id": pack_id, "records": pending, "view": view,
                          "full_bytes": len(full), "view_sha256": sha256(view)})
            pending, parts, size = [], [], 0
    return packs, pending


def alignment(view, target_views):
    if len(view) != CONFIG["score_byte_window"] or any(len(v) != len(view) for v in target_views):
        raise ValueError("All candidate and target views must have identical byte length")
    cx = len(gzip.compress(view, compresslevel=CONFIG["gzip_level"], mtime=0))
    values = []
    for other in target_views:
        cy = len(gzip.compress(other, compresslevel=CONFIG["gzip_level"], mtime=0))
        cxy = len(gzip.compress(view + other, compresslevel=CONFIG["gzip_level"], mtime=0))
        values.append(1 - (cxy - min(cx, cy)) / max(cx, cy))
    if not values:
        raise ValueError("No development views")
    return float(max(values)), float(np.mean(values))


def bucket_weights(bucket, values):
    bucket = np.asarray(bucket, dtype=np.int64)
    positive = np.maximum(np.asarray(values, dtype=float), 0)
    mass = np.bincount(bucket, weights=positive)
    size = np.bincount(bucket)
    if not mass.sum():
        return np.full(len(values), 1 / len(values))
    return mass[bucket] / mass.sum() / size[bucket]


def compute_mixtures(domain, bucket, cr, score_max, score_mean, wrong_max, pack_index, ids, pack_scores):
    n = len(ids)
    domain = np.asarray(domain)
    weights = {"sample_proportional": np.full(n, 1 / n)}
    names, counts = np.unique(domain, return_counts=True)
    count_map = dict(zip(names.tolist(), counts.tolist()))
    weights["uniform_domain"] = np.array([1 / len(names) / count_map[x] for x in domain])
    band = (np.asarray(cr) >= .65) & (np.asarray(cr) <= .80)
    method_status = {m: "available" for m in METHODS}
    if band.any():
        weights["compel_filter"] = band.astype(float) / band.sum()
    else:
        # Zero support is explicit. The trainer must not sample this array.
        weights["compel_filter"] = np.zeros(n)
        method_status["compel_filter"] = "unavailable_empty_support"
    weights["zipmix_static"] = bucket_weights(bucket, score_max)
    weights["wrongtarget_zipmix"] = bucket_weights(bucket, wrong_max)
    # One permutation per original equal-width pack, not one independent
    # permutation for each constituent example with duplicated scores.
    permuted = np.random.default_rng(1405).permutation(pack_scores)
    weights["shuffled_zipmix"] = bucket_weights(bucket, permuted[np.asarray(pack_index)])
    # Grouping changes the published selector's scoring unit. The name remains
    # recognizable, but reports explicitly call this a pack-score adaptation.
    keep = max(1, int(np.ceil(CONFIG["direct_selection_fraction"] * n)))
    ranking = sorted(range(n), key=lambda i: (-float(score_mean[i]), str(ids[i])))
    weights["direct_zipfit"] = np.zeros(n)
    weights["direct_zipfit"][ranking[:keep]] = 1 / keep
    for name, value in weights.items():
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("Invalid mixture values")
        expected = 1 if method_status[name] == "available" else 0
        if not np.isclose(value.sum(), expected):
            raise ValueError("Invalid mixture total")
    return weights, method_status


def save_arrays(path, records, pad_token):
    matrix = np.full((len(records), CONFIG["max_input_length"] - 1), pad_token, dtype=np.int32)
    for i, row in enumerate(records):
        matrix[i, :len(row["tokens"])] = row["tokens"]
    np.savez_compressed(path, input_ids=matrix,
        lengths=np.asarray([len(r["tokens"]) for r in records], dtype=np.int32),
        answer=np.asarray([r["answer"] for r in records], dtype=np.int32),
        answer_token=np.asarray([r["answer_token"] for r in records], dtype=np.int32),
        nchoices=np.asarray([len(r["choices"]) for r in records], dtype=np.int32),
        ids=np.asarray([r["id"] for r in records]),
        question_ids=np.asarray([r["question_id"] for r in records]),
        domain=np.asarray([r["domain"] for r in records]))


def public_pack(pack, role, domain=None):
    return {"pack_id": pack["id"], "role": role, "domain": domain,
            "member_ids": [r["id"] for r in pack["records"]], "members": len(pack["records"]),
            "view_bytes": len(pack["view"]), "full_bytes": pack["full_bytes"],
            "view_sha256": pack["view_sha256"]}


def safe_correlation(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-data", type=Path, default=Path(__file__).parents[1] / "expt_v1" / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output must be new/empty; preserve frozen data")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    parent_manifest = json.loads((args.v1_data / "manifest.json").read_text())
    parent_id = parent_manifest.pop("manifest_id")
    if stable_hash(parent_manifest) != parent_id:
        raise ValueError("Parent data manifest identity changed")
    for name, value in parent_manifest["files"].items():
        if file_hash(args.v1_data / name) != value:
            raise ValueError(f"Frozen parent input changed: {name}")
    records = [json.loads(line) for line in (args.v1_data / "records.jsonl").read_text().splitlines()]
    candidates = [r for r in records if r["role"] == "train"]
    reserved = sorted([r for r in candidates if r["domain"] == "commonsense_qa"], key=lambda r: r["id"])[:CONFIG["wrongtarget_reserved_records"]]
    if len(reserved) != CONFIG["wrongtarget_reserved_records"]:
        raise ValueError("Insufficient wrong-target reserve")
    reserved_ids = {r["id"] for r in reserved}
    candidates = [r for r in candidates if r["id"] not in reserved_ids]
    repo, _, revision = SOURCES["sciq"]
    raw_dev = datasets.load_dataset(repo, revision=revision, split="validation", cache_dir=args.cache_dir)
    true_records, rejection = [], Counter()
    seen = set()
    for raw in raw_dev:
        try:
            record = canonical_record(raw, "sciq", "validation")
        except (ValueError, KeyError) as exc:
            rejection[str(exc)] += 1
            continue
        if record["question_id"] not in seen:
            true_records.append(record)
            seen.add(record["question_id"])
    true_packs, true_tail = make_packs(true_records, "true_development")
    wrong_packs, wrong_tail = make_packs(reserved, "wrong_development")
    nviews = min(CONFIG["alignment_views"], len(true_packs), len(wrong_packs))
    if nviews < CONFIG["minimum_matched_target_views"]:
        raise ValueError("Too few matched real-byte development views")
    true_packs, wrong_packs = true_packs[:nviews], wrong_packs[:nviews]
    true_views = [p["view"] for p in true_packs]
    wrong_views = [p["view"] for p in wrong_packs]
    training_packs, discarded = [], []
    domain_names = sorted({r["domain"] for r in candidates})
    for domain in domain_names:
        packs, tail = make_packs([r for r in candidates if r["domain"] == domain], "training:" + domain)
        if not packs:
            raise ValueError(f"No full real-byte training pack in {domain}")
        for pack in packs:
            pack["domain"] = domain
        training_packs.extend(packs)
        discarded.extend(tail)
    train, pack_index, pack_cr, pack_max, pack_mean, pack_wrong = [], [], [], [], [], []
    for i, pack in enumerate(training_packs):
        raw = pack["view"]
        if len(raw) != CONFIG["score_byte_window"]:
            raise ValueError("Candidate pack width mismatch")
        cr = len(lz4.frame.compress(raw)) / len(raw)
        maximum, mean = alignment(raw, true_views)
        wrong_maximum, _ = alignment(raw, wrong_views)
        pack_cr.append(cr)
        pack_max.append(maximum)
        pack_mean.append(mean)
        pack_wrong.append(wrong_maximum)
        train.extend(pack["records"])
        pack_index.extend([i] * len(pack["records"]))
    pack_index = np.asarray(pack_index, dtype=np.int32)
    pack_cr, pack_max, pack_mean, pack_wrong = map(np.asarray, (pack_cr, pack_max, pack_mean, pack_wrong))
    # Fit quantiles to equally sized packs, then inherit their bins/scores at
    # the one-supervised-token-per-document sampling unit.
    boundaries = np.quantile(pack_cr, np.arange(1, CONFIG["compression_bins"]) / CONFIG["compression_bins"])
    pack_bucket = np.searchsorted(boundaries, pack_cr, side="right")
    bucket = pack_bucket[pack_index]
    domain = np.asarray([domain_names.index(r["domain"]) for r in train])
    cr, maximum, mean, wrong = [values[pack_index] for values in (pack_cr, pack_max, pack_mean, pack_wrong)]
    weights, method_status = compute_mixtures(domain, bucket, cr, maximum, mean, wrong, pack_index,
                                             [r["id"] for r in train], pack_max)
    heldout = [r for r in records if r["role"] in ("target", "retention")]
    protected = true_records + reserved + heldout
    protected_ids = {r["id"] for r in protected}
    protected_q = {r["question_id"] for r in protected}
    protected_grams = set().union(*(content_grams(r) for r in protected))
    exact_hits = sum(r["id"] in protected_ids or r["question_id"] in protected_q for r in train)
    gram_hits = sum(bool(content_grams(r) & protected_grams) for r in train)
    # Reserved wrong-target rows were previously train rows. Newly created
    # near-duplicate overlap requires prospective rebuilding before scoring,
    # not quietly retaining an invalid control.
    if exact_hits or gram_hits:
        raise ValueError(f"Version 2 split invariant failed: exact={exact_hits}, 13gram={gram_hits}")
    save_arrays(args.output / "train.npz", train, parent_manifest["pad_token_id"])
    for name in ("target.npz", "retention.npz"):
        shutil.copyfile(args.v1_data / name, args.output / name)
    shutil.copytree(args.v1_data / "tokenizer", args.output / "tokenizer")
    np.savez_compressed(args.output / "selection.npz", cr=cr, score=maximum, mean_score=mean,
                        wrong_score=wrong, bucket=bucket, domain=domain, pack_index=pack_index,
                        **weights)
    np.savez_compressed(args.output / "pack_scores.npz", cr=pack_cr, maximum=pack_max,
                        mean=pack_mean, wrong_maximum=pack_wrong, bucket=pack_bucket,
                        width=np.full(len(training_packs), CONFIG["score_byte_window"], dtype=np.int32))
    pack_records = [public_pack(p, "train", p["domain"]) for p in training_packs]
    pack_records += [public_pack(p, "true_development", "sciq") for p in true_packs]
    pack_records += [public_pack(p, "wrong_development", "commonsense_qa") for p in wrong_packs]
    atomic_json(args.output / "packs.json", pack_records)
    atomic_json(args.output / "identities.json", {"train": [r["id"] for r in train],
        "wrong_target_reserved": sorted(reserved_ids), "trailing_discarded": [r["id"] for r in discarded],
        "true_development_members": [r["id"] for p in true_packs for r in p["records"]],
        "wrong_development_members": [r["id"] for p in wrong_packs for r in p["records"]]})
    with (args.output / "records.jsonl").open("w") as out:
        for role, rows in (("train", train), ("wrong_target_reserved", reserved),
                           ("true_development", true_records), ("evaluation", heldout)):
            for row in rows:
                out.write(json.dumps({**row, "role": role}, ensure_ascii=False) + "\n")
    files = sorted(p.relative_to(args.output).as_posix() for p in args.output.rglob("*") if p.is_file())
    tv = lambda a, b: float(.5 * np.abs(weights[a] - weights[b]).sum())
    document_lengths = [len(r["content"].encode()) for r in train]
    manifest = {"schema": 2, "version": "expt_v2", "created_at": now(),
        "parent_manifest_id": parent_id, "model": MODEL, "model_revision": MODEL_REVISION,
        "source_revisions": parent_manifest["source_revisions"], "config": CONFIG,
        "methods": list(METHODS), "method_status": method_status, "seeds": list(SEEDS),
        "expected_training_cells": len(METHODS) * len(SEEDS),
        "available_training_cells": sum(v == "available" for v in method_status.values()) * len(SEEDS),
        "label_tokens": parent_manifest["label_tokens"], "pad_token_id": parent_manifest["pad_token_id"],
        "domains": domain_names,
        "counts": {"train": len(train), "training_packs": len(training_packs), "true_views": nviews,
                   "wrong_views": nviews, "wrong_target_reserved": len(reserved),
                   "trailing_training_examples_discarded": len(discarded), "target": 500, "retention": 500},
        "training_domain_counts": dict(Counter(r["domain"] for r in train)),
        "true_validation_conversion_rejections": dict(rejection),
        "postfilter_train_protected_exact_hits": exact_hits, "postfilter_train_protected_13gram_hits": gram_hits,
        "view_bytes": {"min": CONFIG["score_byte_window"], "max": CONFIG["score_byte_window"], "padding_or_repetition": False},
        "pack_member_counts": {"min": min(len(p["records"]) for p in training_packs), "max": max(len(p["records"]) for p in training_packs)},
        "bucket_boundaries": boundaries.tolist(),
        "compression_ratio_range": [float(pack_cr.min()), float(pack_cr.max())],
        "compression_ratio_above_one_count": int((pack_cr > 1).sum()),
        "score_definition": "historical max for ZipMix; published mean aggregation adapted to equal-byte packs for direct selector; candidate_then_development; gzip6; clip negative only for bucket mass",
        "negative_max_scores": int((pack_max < 0).sum()),
        "prior_diagnostics": {"zipmix_vs_wrongtarget_total_variation": tv("zipmix_static", "wrongtarget_zipmix"),
                              "zipmix_vs_shuffled_total_variation": tv("zipmix_static", "shuffled_zipmix"),
                              "zipmix_vs_sample_total_variation": tv("zipmix_static", "sample_proportional"),
                              "pack_cr_vs_member_count_spearman": safe_correlation(pack_cr, [len(p["records"]) for p in training_packs]),
                              "inherited_max_score_vs_document_byte_length_spearman": safe_correlation(maximum, document_lengths)},
        "mixtures": {m: {"status": method_status[m], "positive_examples": int((w > 0).sum()),
                         "effective_examples": float(1 / np.square(w).sum()) if w.sum() else None,
                         "domain_mass": {name: float(w[domain == i].sum()) for i, name in enumerate(domain_names)},
                         "bucket_mass": np.bincount(bucket, weights=w, minlength=CONFIG["compression_bins"]).tolist()}
                     for m, w in weights.items()},
        "files": {name: file_hash(args.output / name) for name in files},
        "preparation_code": {name: file_hash(Path(__file__).parent / name) for name in ("common.py", "prepare_sft.py")},
        "elapsed_seconds": time.monotonic() - start,
        "limitations": ["Pack scores inherited by all member questions; final member may extend past scored prefix",
                        "Packing removes scoring-byte-length variation but source, repetition, and content density remain confounded",
                        "Wrong target is related multiple-choice content, not semantically orthogonal noise",
                        "No ground-truth test result used for selection; same finite test arrays retained from untrained v1",
                        "Small selector shifts may leave a three-seed screen underpowered"],
    }
    manifest["manifest_id"] = stable_hash(manifest)
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"manifest_id": manifest["manifest_id"], "counts": manifest["counts"],
                      "method_status": method_status, "prior_diagnostics": manifest["prior_diagnostics"],
                      "compression_ratio_range": manifest["compression_ratio_range"], "mixtures": manifest["mixtures"]}, indent=2))


if __name__ == "__main__":
    main()
