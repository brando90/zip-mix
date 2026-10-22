#!/usr/bin/env python3
"""Prepare Experiment 04 v2 data: reuse frozen v1 train/dev/eval and add non-overlapping eval_window2.npz."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tokenizers import Tokenizer


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def compute_static_arm_diagnostics(train_npz: Path, domains_map: dict[str, int], target_domains: list[str]) -> dict:
    with np.load(train_npz, allow_pickle=False) as d:
        dom = d["domain"].astype(np.int64)
        bucket = d["bucket"].astype(np.int64)
        score = np.maximum(d["score"].astype(np.float64), 0.0)
    n = len(dom)
    id_to_dom = {v: k for k, v in domains_map.items()}
    target_ids = {domains_map[t] for t in target_domains}
    is_target = np.isin(dom, list(target_ids))

    p_tok = np.full(n, 1.0 / n, dtype=np.float64)

    b_ids = np.unique(bucket)
    b_mass = np.array([score[bucket == k].sum() + 0.001 for k in b_ids], dtype=np.float64)
    b_mass /= b_mass.sum()
    p_zm = np.zeros(n, dtype=np.float64)
    for idx, k in enumerate(b_ids):
        mask = bucket == k
        p_zm[mask] = b_mass[idx] / mask.sum()

    cells = sorted(set(zip(dom.tolist(), bucket.tolist())))
    cell_masks = [(dom == s) & (bucket == b) for s, b in cells]
    cell_counts = np.array([m.sum() for m in cell_masks], dtype=np.float64)
    cell_means = np.array([score[m].mean() for m in cell_masks], dtype=np.float64)
    z_cell = (cell_means - cell_means.mean()) / (cell_means.std(ddof=0) + 1e-12)
    w_cell = cell_counts * np.exp(z_cell)
    w_cell /= w_cell.sum()
    p_db = np.zeros(n, dtype=np.float64)
    for mask, w in zip(cell_masks, w_cell):
        p_db[mask] = w / mask.sum()

    p_dm = np.zeros(n, dtype=np.float64)
    for s in np.unique(dom):
        mask = dom == s
        p_dm[mask] = p_db[mask].sum() / mask.sum()

    perm0 = np.random.default_rng(10000 + 197).permutation(len(cells))
    w_shuf = cell_counts * np.exp(z_cell[perm0])
    w_shuf /= w_shuf.sum()
    p_shuf0 = np.zeros(n, dtype=np.float64)
    for mask, w in zip(cell_masks, w_shuf):
        p_shuf0[mask] = w / mask.sum()

    p_cond = np.zeros(n, dtype=np.float64)
    for mask in cell_masks:
        sc = score[mask]
        if len(np.unique(sc)) <= 1:
            z_in = np.zeros(len(sc), dtype=np.float64)
        else:
            r = rankdata(sc, method="average")
            z_in = (r - 0.5) / len(sc) - 0.5
        w_in = np.exp(np.log(4.0) * z_in)
        p_cond[mask] = mask.mean() * w_in / w_in.sum()

    arms = {
        "token_proportional": p_tok,
        "zipmix_static": p_zm,
        "zipmix_domain_bin": p_db,
        "domain_matched_zipmix": p_dm,
        "shuffled_domain_bin_seed0": p_shuf0,
        "within_domain_conditional": p_cond,
    }
    out = {}
    for name, p in arms.items():
        if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1.0, atol=1e-12):
            raise ValueError(f"Invalid probability vector for {name}")
        out[name] = {
            "target_pubmed_mass": float(p[is_target].sum()),
            "effective_blocks": float(1.0 / (p @ p)),
            "effective_block_fraction": float(1.0 / (p @ p) / n),
            "total_variation_from_token_proportional": float(0.5 * np.abs(p - p_tok).sum()),
            "domain_mass": {id_to_dom[int(s)]: float(p[dom == s].sum()) for s in sorted(id_to_dom)},
            "bucket_mass": [float(p[bucket == b].sum()) for b in range(7)],
        }
    max_dom_diff = max(abs(p_db[dom == s].sum() - p_dm[dom == s].sum()) for s in np.unique(dom))
    max_joint_diff = max(abs(p_cond[m].sum() - m.mean()) for m in cell_masks)
    if max_dom_diff > 1e-12 or max_joint_diff > 1e-12:
        raise ValueError("Invariant check failed on domain_matched_zipmix or within_domain_conditional")
    return {
        "nonempty_domain_bin_cells": len(cells),
        "max_domain_matched_discrepancy": float(max_dom_diff),
        "max_within_domain_conditional_joint_discrepancy": float(max_joint_diff),
        "tv_zipmix_domain_bin_vs_domain_matched": float(0.5 * np.abs(p_db - p_dm).sum()),
        "tv_zipmix_domain_bin_vs_zipmix_static": float(0.5 * np.abs(p_db - p_zm).sum()),
        "tv_within_domain_conditional_vs_token_proportional": float(0.5 * np.abs(p_cond - p_tok).sum()),
        "static_arms": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-data", type=Path,
                        default=Path(__file__).parents[1] / "expt_v1" / "data")
    parser.add_argument("--source-parquet", type=Path,
                        default=Path(__file__).parents[2] / "02_alignment_prior_analysis" / "data_cache" / "pile_texts.parquet")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()

    if (args.output / "manifest.json").exists():
        raise SystemExit("Frozen v2 data exists; use a new directory if changing preparation.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    v1_manifest = json.loads((args.v1_data / "manifest.json").read_text())
    for name, expected_sha in v1_manifest["files"].items():
        actual_sha = digest_file(args.v1_data / name)
        if actual_sha != expected_sha:
            raise ValueError(f"v1 file hash mismatch for {name}: {actual_sha} != {expected_sha}")

    if digest_file(args.source_parquet) != v1_manifest["source_sha256"]:
        raise ValueError("Source parquet hash differs from v1 manifest")

    for name in ("train.npz", "dev.npz", "eval.npz", "tokenizer.json", "split_manifest.json"):
        shutil.copyfile(args.v1_data / name, args.output / name)
        if digest_file(args.output / name) != v1_manifest["files"][name]:
            raise ValueError(f"Copied file hash mismatch: {name}")

    split_manifest = json.loads((args.output / "split_manifest.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.output / "tokenizer.json"))
    frame = pd.read_parquet(args.source_parquet)
    seq_len = int(v1_manifest["seq_len"])
    domain_id = v1_manifest["domains"]
    target_set = set(v1_manifest["target_domains"])

    by_id = {}
    for row in frame.itertuples():
        ident = digest_bytes(normalized(row.text).encode())
        if ident not in by_id:
            raw = row.text.encode()[:8192]
            if len(raw) >= 1024:
                by_id[ident] = raw.decode("utf-8", errors="ignore")

    train_ids = {item["id"] for item in split_manifest["train"]}
    dev_ids = {item["id"] for item in split_manifest["dev"]}
    test_ids = {item["id"] for item in split_manifest["test"]}
    if (train_ids & dev_ids) or (train_ids & test_ids) or (dev_ids & test_ids):
        raise ValueError("Document ID overlap across train/dev/test splits")

    w2_target, w2_broad = [], []
    start_tok = seq_len + 1  # 129: strictly non-overlapping with v1's ids[:129]
    end_tok = 2 * seq_len + 2  # 258: 129 tokens (128 context + 1 next-token target)
    for item in split_manifest["test"]:
        dom_name = item["domain"]
        if dom_name not in domain_id:
            continue
        ids = tokenizer.encode(by_id[item["id"]]).ids
        if len(ids) >= end_tok:
            row = (ids[start_tok:end_tok], domain_id[dom_name], item["id"])
            if dom_name in target_set:
                w2_target.append(row)
            else:
                w2_broad.append(row)

    if not w2_target or not w2_broad:
        raise ValueError("Insufficient window-2 evaluation blocks")

    np.savez_compressed(
        args.output / "eval_window2.npz",
        target_tokens=np.array([r[0] for r in w2_target], dtype=np.int32),
        target_domain=np.array([r[1] for r in w2_target], dtype=np.int32),
        broad_tokens=np.array([r[0] for r in w2_broad], dtype=np.int32),
        broad_domain=np.array([r[1] for r in w2_broad], dtype=np.int32),
    )

    diagnostics = compute_static_arm_diagnostics(
        args.output / "train.npz", domain_id, v1_manifest["target_domains"]
    )
    files = {f.name: digest_file(f) for f in sorted(args.output.iterdir()) if f.is_file() and f.name != "manifest.json"}
    manifest = {
        "schema": 2,
        "version": "expt_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_v1_files": v1_manifest["files"],
        "source_sha256": v1_manifest["source_sha256"],
        "target_domains": v1_manifest["target_domains"],
        "domains": domain_id,
        "train_documents": v1_manifest["train_documents"],
        "train_blocks": v1_manifest["train_blocks"],
        "dev_blocks": v1_manifest["dev_blocks"],
        "target_eval_blocks": v1_manifest["target_eval_blocks"],
        "broad_eval_blocks": v1_manifest["broad_eval_blocks"],
        "fresh_window2_target_eval_blocks": len(w2_target),
        "fresh_window2_broad_eval_blocks": len(w2_broad),
        "fresh_window2_token_slice": [start_tok, end_tok],
        "vocab_size": v1_manifest["vocab_size"],
        "seq_len": seq_len,
        "selector_diagnostics": diagnostics,
        "seconds": time.time() - started,
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (Path(__file__).parent / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "train_blocks": manifest["train_blocks"],
        "fresh_window2_target_eval_blocks": manifest["fresh_window2_target_eval_blocks"],
        "fresh_window2_broad_eval_blocks": manifest["fresh_window2_broad_eval_blocks"],
        "nonempty_domain_bin_cells": diagnostics["nonempty_domain_bin_cells"],
        "pubmed_mass": {k: v["target_pubmed_mass"] for k, v in diagnostics["static_arms"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
