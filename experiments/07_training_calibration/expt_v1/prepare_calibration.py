#!/usr/bin/env python3
"""Freeze Experiment 07's fresh calibration and Stage B data on CPU.

1. Rebuild Experiment 05's version 1 and version 2 selections from the pinned
   public revisions and verify their identity files against the committed
   parent manifests, so "previously used" is exact, not approximate.
2. Rebuild the version 2 training pool, its equal-byte pack scores and the
   NEXT_STAGE_DESIGN.md probabilities; verify they reproduce the committed
   diagnostics receipt.
3. Allocate fresh, mutually disjoint calibration and Stage B holdout records
   that no earlier experiment trained on, scored against or evaluated.

No model is loaded or called. Output directories must be new or empty.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import shutil
import time

import datasets
from huggingface_hub import HfApi
import lz4.frame
import numpy as np
from transformers import AutoTokenizer

from calib_common import (ARC_FRESH_SPLIT, EVALUATION, EXP05, EXPERIMENTS, HOME, LABELS,
                          MODEL, MODEL_REVISION, SOURCES, SPLIT, STAGE_A, STAGE_B, TRAIN,
                          V1_CONFIG, V1_MANIFEST, V2_CONFIG, V2_MANIFEST, Protected,
                          admission_checks, assign_roles, atomic_json, canonical_record,
                          content_grams, file_hash, load_module, now, stable_hash,
                          stage_b_probabilities)

V1 = load_module(EXP05 / "expt_v1" / "prepare_sft.py", "exp05_v1_prepare", EXP05 / "expt_v1" / "common.py")
V2 = load_module(EXP05 / "expt_v2" / "prepare_sft.py", "exp05_v2_prepare", EXP05 / "expt_v2" / "common.py")
DIAGNOSTICS = EXPERIMENTS / "04_validation_guided_zipmix" / "expt_v1" / "results" / "next_stage_diagnostics.json"


def identity_digest(value) -> str:
    """Hash exactly as atomic_json would write the parent identities.json."""
    from calib_common import sha256
    return sha256((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_sources(cache_dir):
    api = HfApi()
    require(api.model_info(MODEL, revision=MODEL_REVISION).sha == MODEL_REVISION, "Model revision unresolved")
    loaded = {}
    for domain, (repo, config, revision) in SOURCES.items():
        require(api.dataset_info(repo, revision=revision).sha == revision, f"Dataset revision unresolved: {repo}")
        kwargs = dict(path=repo, revision=revision, cache_dir=cache_dir)
        if config is not None:
            kwargs["name"] = config
        loaded[domain] = datasets.load_dataset(**kwargs)
    return loaded


def rebuild_v1(loaded, tokenizer, label_tokens):
    """Experiment 05 v1 selection logic, minus scoring; returns provenance."""
    pools, held = {}, {}
    for domain, data in loaded.items():
        pools[domain], _ = V1.convert(data["train"], domain, "train")
    for role, (domain, split) in {"alignment": ("sciq", "validation"), "target": ("sciq", "test"),
                                  "retention": ("commonsense_qa", "validation")}.items():
        held[role], _ = V1.convert(loaded[domain][split], domain, split)
    protected = Protected([r for rows in held.values() for r in rows])
    train, eligible, seen_questions = [], {}, set()
    for domain in sorted(pools):
        candidates = []
        for record in pools[domain]:
            if protected.hits(record) or record["question_id"] in seen_questions:
                continue
            encoded = V1.encode_record(record, tokenizer, label_tokens)
            if encoded is None:
                continue
            seen_questions.add(record["question_id"])
            candidates.append(encoded)
        eligible[domain] = candidates
        train.extend(candidates[:V1_CONFIG["training_cap_per_domain"]])
    eval_rows, seen = {}, Protected()
    for role in ("alignment", "target", "retention"):
        candidates, role_questions = [], set()
        for row in held[role]:
            if row["question_id"] in role_questions or seen.hits(row):
                continue
            encoded = V1.encode_record(row, tokenizer, label_tokens)
            if encoded is None:
                continue
            role_questions.add(row["question_id"])
            candidates.append(encoded)
        cap = V1_CONFIG["alignment_views"] if role == "alignment" else V1_CONFIG["evaluation_cap_per_split"]
        eval_rows[role] = candidates[:cap]
        seen.add(candidates)
    train = sorted(train, key=lambda r: (r["domain"], r["id"]))
    identities = {"train": [r["id"] for r in train], **{k: [r["id"] for r in v] for k, v in eval_rows.items()}}
    manifest = json.loads(V1_MANIFEST.read_text())
    require(identity_digest(identities) == manifest["files"]["identities.json"],
            "Experiment 05 v1 identities did not reproduce; refuse to define fresh records")
    return {"train": train, "eligible": eligible, "held": held, "eval": eval_rows,
            "seen_questions": seen_questions, "identities_sha256": identity_digest(identities)}


def rebuild_v2(v1, loaded):
    """Experiment 05 v2 packs and scores; verifies identities and diagnostics."""
    config = V2_CONFIG
    candidates = list(v1["train"])
    reserved = sorted([r for r in candidates if r["domain"] == "commonsense_qa"],
                      key=lambda r: r["id"])[:config["wrongtarget_reserved_records"]]
    reserved_ids = {r["id"] for r in reserved}
    candidates = [r for r in candidates if r["id"] not in reserved_ids]
    true_records, seen = [], set()
    for raw in loaded["sciq"]["validation"]:
        try:
            record = canonical_record(raw, "sciq", "validation")
        except (ValueError, KeyError):
            continue
        if record["question_id"] not in seen:
            true_records.append(record)
            seen.add(record["question_id"])
    true_packs, _ = V2.make_packs(true_records, "true_development")
    wrong_packs, _ = V2.make_packs(reserved, "wrong_development")
    nviews = min(config["alignment_views"], len(true_packs), len(wrong_packs))
    true_packs, wrong_packs = true_packs[:nviews], wrong_packs[:nviews]
    true_views, wrong_views = [p["view"] for p in true_packs], [p["view"] for p in wrong_packs]
    domains = sorted({r["domain"] for r in candidates})
    packs, discarded = [], []
    for domain in domains:
        made, tail = V2.make_packs([r for r in candidates if r["domain"] == domain], "training:" + domain)
        for pack in made:
            pack["domain"] = domain
        packs.extend(made)
        discarded.extend(tail)
    train, pack_index, cr, maximum, wrong = [], [], [], [], []
    for i, pack in enumerate(packs):
        cr.append(len(lz4.frame.compress(pack["view"])) / len(pack["view"]))
        maximum.append(V2.alignment(pack["view"], true_views)[0])
        wrong.append(V2.alignment(pack["view"], wrong_views)[0])
        train.extend(pack["records"])
        pack_index.extend([i] * len(pack["records"]))
    cr, maximum, wrong, pack_index = map(np.asarray, (cr, maximum, wrong, pack_index))
    boundaries = np.quantile(cr, np.arange(1, config["compression_bins"]) / config["compression_bins"])
    pack_bucket = np.searchsorted(boundaries, cr, side="right")
    identities = {"train": [r["id"] for r in train], "wrong_target_reserved": sorted(reserved_ids),
                  "trailing_discarded": [r["id"] for r in discarded],
                  "true_development_members": [r["id"] for p in true_packs for r in p["records"]],
                  "wrong_development_members": [r["id"] for p in wrong_packs for r in p["records"]]}
    manifest = json.loads(V2_MANIFEST.read_text())
    require(identity_digest(identities) == manifest["files"]["identities.json"],
            "Experiment 05 v2 identities did not reproduce")
    require(np.allclose(boundaries, manifest["bucket_boundaries"], rtol=0, atol=1e-15), "v2 compression bins changed")
    source = np.asarray([domains.index(r["domain"]) for r in train])
    bucket = pack_bucket[pack_index]
    probabilities, masks = stage_b_probabilities(source, bucket, pack_index, maximum - wrong)
    lengths = np.asarray([len(r["tokens"]) for r in train])
    receipt = json.loads(DIAGNOSTICS.read_text())
    reproduced = {"aligned": probabilities["conditional_signed_rank"], "uniform": probabilities["uniform_stratum"],
                  "shuffled": probabilities["permuted_stratum_rank"]}
    for name, p in reproduced.items():
        expected = receipt["metrics"][name]
        require(abs(1 / (p @ p) - expected["effective_sample_size"]) < 1e-6, f"Stage B {name} ESS did not reproduce")
        require(abs(p @ lengths - expected["expected_input_tokens_per_question"]) < 1e-9, f"Stage B {name} length did not reproduce")
    for key, value in receipt["total_variation"].items():
        a, b = key.split("_vs_")
        require(abs(np.abs(reproduced[a] - reproduced[b]).sum() / 2 - value) < 1e-9, f"Stage B {key} did not reproduce")
    admission = admission_checks(probabilities, masks, source, bucket)
    require(admission["passed"], "Stage B admission gates failed; do not train")
    used = Protected(true_records + reserved + train)
    return {"train": train, "domains": domains, "source": source, "bucket": bucket, "pack_index": pack_index,
            "pack_scores": {"cr": cr, "maximum": maximum, "wrong_maximum": wrong, "bucket": pack_bucket},
            "probabilities": probabilities, "admission": admission, "dev_records": true_records,
            "reserved": reserved, "identities_sha256": identity_digest(identities), "used": used,
            "pack_count": len(packs), "strata": len(masks)}


def fresh_pools(v1, loaded, tokenizer, label_tokens):
    cap = V1_CONFIG["training_cap_per_domain"]
    fresh = {d: list(rows[cap:]) for d, rows in v1["eligible"].items()}
    arc, _ = V1.convert(loaded["arc_challenge"][ARC_FRESH_SPLIT], "arc_challenge", ARC_FRESH_SPLIT)
    seen, extra = set(v1["seen_questions"]), []
    for record in arc:
        if record["question_id"] in seen:
            continue
        encoded = V1.encode_record(record, tokenizer, label_tokens)
        if encoded is not None:
            seen.add(record["question_id"])
            extra.append(encoded)
    fresh["arc_challenge"] = fresh["arc_challenge"] + extra
    return fresh


def score_calibration_pool(train, dev_records):
    dev_packs, _ = V2.make_packs(dev_records, "calibration_development")
    require(len(dev_packs) >= V2_CONFIG["alignment_views"], "Too few calibration development views")
    views = [p["view"] for p in dev_packs[:V2_CONFIG["alignment_views"]]]
    packs = []
    for domain in sorted({r["domain"] for r in train}):
        made, _ = V2.make_packs([r for r in train if r["domain"] == domain], "calibration:" + domain)
        packs.extend(made)
    cr = np.asarray([len(lz4.frame.compress(p["view"])) / len(p["view"]) for p in packs])
    maximum = np.asarray([V2.alignment(p["view"], views)[0] for p in packs])
    domain = np.asarray([p["records"][0]["domain"] for p in packs])
    return {"cr": cr, "maximum": maximum, "domain": domain, "packs": len(packs),
            "views": len(views)}


def write_split(directory, arrays, pad_token, tokenizer_dir, records_by_role, extra_npz, manifest):
    directory.mkdir(parents=True, exist_ok=False)
    for name, rows in arrays.items():
        V1.save_arrays(directory / f"{name}.npz", rows, pad_token)
    for name, values in extra_npz.items():
        np.savez_compressed(directory / f"{name}.npz", **values)
    shutil.copytree(tokenizer_dir, directory / "tokenizer")
    with (directory / "records.jsonl").open("w") as out:
        for role, rows in records_by_role.items():
            for row in rows:
                out.write(json.dumps({**row, "role": role}, ensure_ascii=False) + "\n")
    atomic_json(directory / "identities.json", {role: [r["id"] for r in rows] for role, rows in records_by_role.items()})
    files = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())
    manifest["files"] = {name: file_hash(directory / name) for name in files}
    manifest["manifest_id"] = stable_hash(manifest)
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def leakage_audit(roles, guards):
    """Recount every pairwise exact/13-gram overlap after allocation."""
    names = list(roles)
    counts = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            guard = Protected(roles[b])
            counts[f"{a}|{b}"] = sum(guard.hits(r) for r in roles[a])
    for label, guard in guards.items():
        for name, rows in roles.items():
            counts[f"{name}|{label}"] = sum(guard.hits(r) for r in rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HOME / "data")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output must be new/empty; frozen data is never overwritten")
    start = time.monotonic()
    loaded = load_sources(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, cache_dir=args.cache_dir)
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    label_tokens = [tokenizer.encode(" " + x, add_special_tokens=False)[0] for x in LABELS]
    parent_v1 = json.loads(V1_MANIFEST.read_text())
    require(label_tokens == parent_v1["label_tokens"] and pad_token == parent_v1["pad_token_id"], "Tokenizer changed")
    v1 = rebuild_v1(loaded, tokenizer, label_tokens)
    v2 = rebuild_v2(v1, loaded)
    # Everything any earlier experiment trained on, scored against or protected.
    prior = Protected(v1["train"] + [r for rows in v1["held"].values() for r in rows])
    prior.add(v2["dev_records"] + v2["reserved"])
    prior_snapshot = Protected(v1["train"] + [r for rows in v1["held"].values() for r in rows] + v2["dev_records"])
    fresh = fresh_pools(v1, loaded, tokenizer, label_tokens)
    roles, exclusions = assign_roles(fresh, prior)
    audit = leakage_audit({k: v for k, v in roles.items()}, {"experiment05_06_inputs": prior_snapshot})
    require(not any(audit.values()), f"Leakage audit failed: { {k: v for k, v in audit.items() if v} }")
    v2_train_guard = Protected(v2["train"])
    holdout_vs_v2 = {r: sum(v2_train_guard.hits(x) for x in roles[r]) for r in ("holdout_target", "holdout_retention", "calibration_target", "calibration_retention")}
    require(not any(holdout_vs_v2.values()), "Fresh evaluation overlaps the Stage B training pool")
    scores = score_calibration_pool(roles["calibration_train"], roles["calibration_dev_views"])
    tokenizer_dir = args.output / ".tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    shared = {"schema": 1, "experiment": "07_training_calibration/expt_v1", "created_at": now(),
              "model": MODEL, "model_revision": MODEL_REVISION,
              "source_revisions": {d: {"repository": r, "config": c, "revision": v} for d, (r, c, v) in SOURCES.items()},
              "arc_fresh_split": ARC_FRESH_SPLIT, "label_tokens": label_tokens, "pad_token_id": pad_token,
              "train_config": TRAIN, "split_config": SPLIT, "stage_a": STAGE_A, "stage_b": STAGE_B,
              "parents": {"exp05_v1_manifest_id": parent_v1["manifest_id"],
                          "exp05_v1_identities_sha256": v1["identities_sha256"],
                          "exp05_v2_manifest_id": json.loads(V2_MANIFEST.read_text())["manifest_id"],
                          "exp05_v2_identities_sha256": v2["identities_sha256"],
                          "identities_reproduced": True, "next_stage_diagnostics_reproduced": True},
              "leakage_rule": "exact normalized question/content identity or any shared 13-word contiguous question/choice segment; wrappers and answer labels never scanned",
              "leakage_audit_pairwise_hits": audit, "fresh_evaluation_vs_stage_b_training_hits": holdout_vs_v2,
              "allocation_exclusions": exclusions,
              "fresh_pool_sizes": {d: len(v) for d, v in fresh.items()},
              "preparation_code": {n: file_hash(HOME / n) for n in ("calib_common.py", "prepare_calibration.py")},
              "runtime": {"python": platform.python_version(), "datasets": datasets.__version__, "numpy": np.__version__}}
    calibration_records = {k: roles[k] for k in ("calibration_train", "calibration_target", "calibration_retention", "calibration_dev_views")}
    calibration = write_split(args.output / "calibration",
        {"train": roles["calibration_train"], "calibration_target": roles["calibration_target"],
         "calibration_retention": roles["calibration_retention"]},
        pad_token, tokenizer_dir, calibration_records,
        {"pack_scores": {"cr": scores["cr"], "maximum": scores["maximum"], "domain": scores["domain"]}},
        {**shared, "stage": "calibration", "evaluation_sets": EVALUATION["calibration"],
         "counts": {k: len(v) for k, v in calibration_records.items()},
         "training_domain_counts": dict(Counter(r["domain"] for r in roles["calibration_train"])),
         "pool_diagnostics": {"calibration_packs": scores["packs"], "development_views": scores["views"],
                              "calibration_cr_range": [float(scores["cr"].min()), float(scores["cr"].max())],
                              "calibration_alignment_mean": float(scores["maximum"].mean()),
                              "exp05_v2_cr_range": [float(v2["pack_scores"]["cr"].min()), float(v2["pack_scores"]["cr"].max())],
                              "exp05_v2_alignment_mean": float(v2["pack_scores"]["maximum"].mean())},
         "information_boundary": "Stage B holdout records are absent from this directory"})
    selection = {"domain": v2["source"], "bucket": v2["bucket"], "pack_index": v2["pack_index"],
                 **v2["probabilities"]}
    stage_b_records = {"train": v2["train"], **{k: roles[k] for k in ("holdout_target", "holdout_retention", "calibration_target", "calibration_retention")}}
    stage_b = write_split(args.output / "stage_b",
        {"train": v2["train"], **{k: roles[k] for k in ("holdout_target", "holdout_retention", "calibration_target", "calibration_retention")}},
        pad_token, tokenizer_dir, stage_b_records,
        {"selection": selection, "pack_scores": v2["pack_scores"]},
        {**shared, "stage": "stage_b", "evaluation_sets": EVALUATION["stage_b"],
         "counts": {k: len(v) for k, v in stage_b_records.items()}, "domains": v2["domains"],
         "training_pool": "Experiment 05 v2 pool rebuilt from pinned sources; identities verified against the committed manifest",
         "training_packs": v2["pack_count"], "strata": v2["strata"], "admission": v2["admission"],
         "recipe_source": "Stage A decision.json; trainer refuses Stage B without a passed decision"})
    shutil.rmtree(tokenizer_dir)
    summary = {"calibration_manifest_id": calibration["manifest_id"], "stage_b_manifest_id": stage_b["manifest_id"],
               "calibration_counts": calibration["counts"], "stage_b_counts": stage_b["counts"],
               "admission": v2["admission"], "elapsed_seconds": time.monotonic() - start}
    for name, manifest in (("calibration", calibration), ("stage_b", stage_b)):
        atomic_json(HOME / f"{name}_manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
