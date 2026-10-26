#!/usr/bin/env python3
"""Prepare Experiment 07 data partitions and probability vectors without reading prior benchmark outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import datasets
import numpy as np
from transformers import AutoTokenizer

from common import (
    ALL_STAGE2_METHODS,
    CALIBRATION_LRS,
    CALIBRATION_SEEDS,
    CONDITIONAL_METHODS,
    CONDITIONAL_SEEDS,
    CONFIG,
    LABELS,
    MODEL,
    MODEL_REVISION,
    PARENT_EXP06_MANIFEST_ID,
    PARENT_V1_MANIFEST_ID,
    PARENT_V2_MANIFEST_ID,
    SELECTOR_METHODS,
    SELECTOR_SEEDS,
    SOURCES,
    atomic_json,
    canonical_record,
    content_grams,
    file_hash,
    now,
    stable_hash,
    within_stratum_conditional_weights,
)


def convert(rows, domain: str, split: str) -> tuple[list[dict], dict]:
    out, rejected = [], Counter()
    for row in rows:
        try:
            out.append(canonical_record(row, domain, split))
        except (ValueError, KeyError) as exc:
            rejected[str(exc)] += 1
    return sorted(out, key=lambda r: r["id"]), dict(rejected)


def encode_record(row: dict, tokenizer, label_tokens: list[int]) -> dict | None:
    tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
    for i in range(len(row["choices"])):
        full = tokenizer.encode(row["prompt"] + " " + LABELS[i], add_special_tokens=False)
        if full != tokens + [label_tokens[i]]:
            raise ValueError("Prompt/answer tokenization boundary changed")
    if len(tokens) + 1 > CONFIG["max_input_length"]:
        return None
    return {**row, "tokens": tokens, "answer_token": label_tokens[row["answer"]]}


def save_arrays(path: Path, records: list[dict], pad_token: int) -> None:
    matrix = np.full((len(records), CONFIG["max_input_length"] - 1), pad_token, dtype=np.int32)
    for i, row in enumerate(records):
        matrix[i, :len(row["tokens"])] = row["tokens"]
    np.savez_compressed(
        path,
        input_ids=matrix,
        lengths=np.asarray([len(r["tokens"]) for r in records], dtype=np.int32),
        answer=np.asarray([r["answer"] for r in records], dtype=np.int32),
        answer_token=np.asarray([r["answer_token"] for r in records], dtype=np.int32),
        nchoices=np.asarray([len(r["choices"]) for r in records], dtype=np.int32),
        ids=np.asarray([r["id"] for r in records]),
        question_ids=np.asarray([r["question_id"] for r in records]),
        domain=np.asarray([r["domain"] for r in records]),
    )


def verify_manifest(path: Path, expected_id: str) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    actual_id = manifest.pop("manifest_id")
    if actual_id != expected_id or stable_hash(manifest) != actual_id:
        raise ValueError(f"Manifest verification failed for {path}")
    manifest["manifest_id"] = actual_id
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root_expts = Path(__file__).resolve().parents[2]
    parser.add_argument("--v1-data", type=Path, default=root_expts / "05_validation_guided_sft/expt_v1/data")
    parser.add_argument("--v2-data", type=Path, default=root_expts / "05_validation_guided_sft/expt_v2/data")
    parser.add_argument("--exp06-data", type=Path, default=root_expts / "06_contrastive_validation_zipmix/expt_v1/data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output must be new and empty; do not overwrite frozen data")

    start = time.monotonic()
    v1_manifest = verify_manifest(args.v1_data, PARENT_V1_MANIFEST_ID)
    v2_manifest = verify_manifest(args.v2_data, PARENT_V2_MANIFEST_ID)
    exp06_manifest = verify_manifest(args.exp06_data, PARENT_EXP06_MANIFEST_ID)

    # Verify required parent files
    for name in ("identities.json", "records.jsonl"):
        if file_hash(args.v1_data / name) != v1_manifest["files"][name]:
            raise ValueError(f"v1 file changed: {name}")
        if file_hash(args.v2_data / name) != v2_manifest["files"][name]:
            raise ValueError(f"v2 file changed: {name}")
    for name in ("train.npz", "target.npz", "retention.npz", "pack_scores.npz", "selection.npz"):
        if file_hash(args.v2_data / name) != v2_manifest["files"][name]:
            raise ValueError(f"v2 array changed: {name}")
    if file_hash(args.exp06_data / "selection.npz") != exp06_manifest["files"]["selection.npz"]:
        raise ValueError("exp06 selection.npz changed")

    # Collect all record identities used in v1 and v2
    id1 = json.loads((args.v1_data / "identities.json").read_text())
    id2 = json.loads((args.v2_data / "identities.json").read_text())
    used_ids = set().union(*(set(v) for v in id1.values()), *(set(v) for v in id2.values()))

    recs1 = {r["id"]: r for line in (args.v1_data / "records.jsonl").read_text().splitlines() for r in [json.loads(line)]}
    recs2 = {r["id"]: r for line in (args.v2_data / "records.jsonl").read_text().splitlines() for r in [json.loads(line)]}
    all_prior_by_id = {**recs1, **recs2}
    used_records = [all_prior_by_id[i] for i in used_ids if i in all_prior_by_id]
    if len(used_records) != len(used_ids):
        raise ValueError("Missing used record provenance in v1/v2 records.jsonl")

    protected_ids = set(used_ids)
    protected_qids = {r["question_id"] for r in used_records}
    protected_grams = set().union(*(content_grams(r) for r in used_records))

    tokenizer = AutoTokenizer.from_pretrained(args.v2_data / "tokenizer")
    label_tokens = v2_manifest["label_tokens"]
    pad_token = v2_manifest["pad_token_id"]

    loaded = {}
    for domain in ("sciq", "commonsense_qa"):
        repo, cfg, rev = SOURCES[domain]
        kwargs = dict(path=repo, revision=rev, cache_dir=str(args.cache_dir) if args.cache_dir else None)
        if cfg is not None:
            kwargs["name"] = cfg
        loaded[domain] = datasets.load_dataset(**kwargs)

    # Priority order for fresh splits:
    # 1. target (sciq:validation, 500)
    # 2. retention (commonsense_qa:validation, 500)
    # 3. test_holdout2 (sciq:test, 490)
    # 4. cal_target (sciq:train, 500)
    # 5. cal_retention (commonsense_qa:train, 500)
    role_specs = [
        ("target", "sciq", "validation", CONFIG["evaluation_cap_per_split"]),
        ("retention", "commonsense_qa", "validation", CONFIG["evaluation_cap_per_split"]),
        ("test_holdout2", "sciq", "test", CONFIG["test_holdout2_count"]),
        ("cal_target", "sciq", "train", CONFIG["evaluation_cap_per_split"]),
        ("cal_retention", "commonsense_qa", "train", CONFIG["evaluation_cap_per_split"]),
    ]

    partition_rows = {}
    partition_stats = {}
    for role, domain, split, cap in role_specs:
        raw_rows, rejected = convert(loaded[domain][split], domain, split)
        eligible = []
        exclusions = Counter()
        role_qids = set()
        for row in raw_rows:
            if row["id"] in protected_ids or row["question_id"] in protected_qids:
                exclusions["prior_or_cross_role_exact"] += 1
                continue
            if row["question_id"] in role_qids:
                exclusions["within_role_duplicate_question"] += 1
                continue
            grams = content_grams(row)
            if grams & protected_grams:
                exclusions["prior_or_cross_role_13gram"] += 1
                continue
            encoded = encode_record(row, tokenizer, label_tokens)
            if encoded is None:
                exclusions["overlength"] += 1
                continue
            role_qids.add(row["question_id"])
            eligible.append(encoded)
        if len(eligible) < cap:
            raise ValueError(f"Insufficient clean eligible items for {role}: {len(eligible)} < {cap}")
        chosen = eligible[:cap]
        partition_rows[role] = chosen
        # Protect all chosen records (and for official val/test splits, all eligible records)
        protect_pool = eligible if split != "train" else chosen
        for row in protect_pool:
            protected_ids.add(row["id"])
            protected_qids.add(row["question_id"])
            protected_grams.update(content_grams(row))
        partition_stats[role] = {
            "domain": domain,
            "official_split": split,
            "raw_valid": len(raw_rows),
            "clean_eligible": len(eligible),
            "chosen": len(chosen),
            "conversion_rejections": rejected,
            "exclusions": dict(exclusions),
        }

    # Verify zero exact or 13-gram overlap between train.npz and all 5 new splits
    train_records = [r for r in recs2.values() if r["role"] == "train"]
    new_eval_records = [r for rows in partition_rows.values() for r in rows]
    new_eval_ids = {r["id"] for r in new_eval_records}
    new_eval_qids = {r["question_id"] for r in new_eval_records}
    new_eval_grams = set().union(*(content_grams(r) for r in new_eval_records))
    exact_hits = sum(r["id"] in new_eval_ids or r["question_id"] in new_eval_qids for r in train_records)
    gram_hits = sum(bool(content_grams(r) & new_eval_grams) for r in train_records)
    if exact_hits or gram_hits:
        raise ValueError(f"Contamination invariant failed: exact={exact_hits}, 13gram={gram_hits}")

    # Build selection.npz combining within-stratum conditional methods + cross-stratum methods
    with np.load(args.v2_data / "selection.npz", allow_pickle=False) as s_v2:
        domain = s_v2["domain"]
        bucket = s_v2["bucket"]
        pack_index = s_v2["pack_index"]
        v2_weights = {m: s_v2[m].astype(np.float64) for m in ("zipmix_static", "shuffled_zipmix", "wrongtarget_zipmix", "direct_zipfit")}
    with np.load(args.v2_data / "pack_scores.npz", allow_pickle=False) as ps_v2:
        true_max = ps_v2["maximum"]
        wrong_max = ps_v2["wrong_maximum"]
    with np.load(args.exp06_data / "selection.npz", allow_pickle=False) as s_06:
        exp06_weights = {m: s_06[m].astype(np.float64) for m in ("contrastive_zipmix", "source_matched")}
    with np.load(args.v2_data / "train.npz", allow_pickle=False) as tr_v2:
        lengths = tr_v2["lengths"]

    cond_weights, cond_diagnostics = within_stratum_conditional_weights(
        domain, bucket, pack_index, true_max, wrong_max,
        permutation_seed=CONFIG["conditional_permutation_seed"],
    )
    all_weights = {**cond_weights, **v2_weights, **exp06_weights}
    for m in ALL_STAGE2_METHODS:
        p = all_weights[m]
        if len(p) != len(domain) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1.0, atol=1e-12):
            raise ValueError(f"Invalid probability vector for {m}")

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.v2_data / "train.npz", args.output / "train.npz")
    shutil.copyfile(args.v2_data / "pack_scores.npz", args.output / "pack_scores.npz")
    shutil.copyfile(args.v2_data / "target.npz", args.output / "reused_target.npz")
    shutil.copyfile(args.v2_data / "retention.npz", args.output / "reused_retention.npz")
    shutil.copytree(args.v2_data / "tokenizer", args.output / "tokenizer")

    for role, rows in partition_rows.items():
        save_arrays(args.output / f"{role}.npz", rows, pad_token)

    np.savez_compressed(
        args.output / "selection.npz",
        domain=domain,
        bucket=bucket,
        pack_index=pack_index,
        **all_weights,
    )

    atomic_json(args.output / "identities.json", {
        role: [r["id"] for r in rows] for role, rows in partition_rows.items()
    })

    mixtures = {}
    for m in ALL_STAGE2_METHODS:
        p = all_weights[m]
        mixtures[m] = {
            "nonzero_examples": int((p > 0).sum()),
            "effective_examples": float(1.0 / (p @ p)),
            "effective_fraction": float((1.0 / (p @ p)) / len(p)),
            "expected_input_tokens_per_example": float(p @ lengths),
            "source_mass": {name: float(p[domain == i].sum()) for i, name in enumerate(v2_manifest["domains"])},
            "bucket_mass": np.bincount(bucket, weights=p, minlength=CONFIG["compression_bins"]).tolist(),
        }

    rel_files = sorted(
        p.relative_to(args.output).as_posix()
        for p in args.output.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )
    manifest = {
        "schema": 4,
        "version": "exp07_v1",
        "experiment": "07_calibrated_conditional_sft_zipmix",
        "created_at": now(),
        "parents": {
            "v1_manifest_id": PARENT_V1_MANIFEST_ID,
            "v2_manifest_id": PARENT_V2_MANIFEST_ID,
            "exp06_manifest_id": PARENT_EXP06_MANIFEST_ID,
        },
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "config": CONFIG,
        "calibration_grid": {
            "learning_rates": CALIBRATION_LRS,
            "seeds": list(CALIBRATION_SEEDS),
            "expected_cells": len(CALIBRATION_LRS) * len(CALIBRATION_SEEDS),
            "sampling_method": "stratum_uniform",
        },
        "stage2_conditional": {
            "methods": list(CONDITIONAL_METHODS),
            "seeds": list(CONDITIONAL_SEEDS),
            "expected_cells": len(CONDITIONAL_METHODS) * len(CONDITIONAL_SEEDS),
            "diagnostics": cond_diagnostics,
        },
        "stage2_selectors": {
            "methods": list(SELECTOR_METHODS),
            "seeds": list(SELECTOR_SEEDS),
            "expected_cells": len(SELECTOR_METHODS) * len(SELECTOR_SEEDS),
        },
        "domains": v2_manifest["domains"],
        "pad_token_id": pad_token,
        "label_tokens": label_tokens,
        "counts": {
            "train": int(len(domain)),
            "training_packs": int(len(true_max)),
            **{role: len(rows) for role, rows in partition_rows.items()},
            "reused_target": 500,
            "reused_retention": 500,
        },
        "partition_stats": partition_stats,
        "postfilter_train_new_eval_exact_hits": exact_hits,
        "postfilter_train_new_eval_13gram_hits": gram_hits,
        "mixtures": mixtures,
        "preparation_code": {
            name: file_hash(Path(__file__).parent / name)
            for name in ("common.py", "prepare_exp07.py")
        },
        "files": {name: file_hash(args.output / name) for name in rel_files},
        "elapsed_seconds": time.monotonic() - start,
    }
    manifest["manifest_id"] = stable_hash(manifest)
    atomic_json(args.output / "manifest.json", manifest)
    atomic_json(Path(__file__).parent / "data_manifest.json", manifest)
    print(json.dumps({
        "manifest_id": manifest["manifest_id"],
        "counts": manifest["counts"],
        "conditional_diagnostics": cond_diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
