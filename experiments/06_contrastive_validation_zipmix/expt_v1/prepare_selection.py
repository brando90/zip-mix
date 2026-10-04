#!/usr/bin/env python3
"""Freeze three contrastive selectors from the existing training/dev scores.

No models, outcome metrics, new examples, or benchmark-driven choices are used.
Held-out arrays are copied byte-for-byte without loading their contents.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from common import (CONFIG, METHODS, MODEL, MODEL_REVISION, SEEDS,
                    atomic_json, file_hash, now, stable_hash)

PARENT_MANIFEST_ID = "0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44"


def bucket_prior(bucket, values):
    bucket, values = np.asarray(bucket), np.asarray(values, dtype=float)
    if bucket.ndim != 1 or values.shape != bucket.shape or not len(bucket):
        raise ValueError("Invalid selector dimensions")
    if bucket.dtype.kind not in 'iu' or (bucket < 0).any():
        raise ValueError("Invalid bin identifiers")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Nonfinite or negative selector mass")
    mass = np.bincount(bucket, weights=values)
    count = np.bincount(bucket)
    if mass.sum() == 0:
        return np.zeros(len(bucket))
    return mass[bucket] / mass.sum() / count[bucket]


def selectors(domain, bucket, pack_index, true_max, wrong_max, permutation_seed):
    domain, bucket, pack_index = map(np.asarray, (domain, bucket, pack_index))
    true_max, wrong_max = np.asarray(true_max, dtype=float), np.asarray(wrong_max, dtype=float)
    if domain.ndim != 1 or not len(domain) or domain.shape != bucket.shape or domain.shape != pack_index.shape:
        raise ValueError("Invalid example arrays")
    if pack_index.dtype.kind not in 'iu' or (pack_index < 0).any():
        raise ValueError("Invalid pack indexes")
    if true_max.ndim != 1 or true_max.shape != wrong_max.shape or not len(true_max):
        raise ValueError("Invalid pack scores")
    if not np.isfinite(true_max).all() or not np.isfinite(wrong_max).all():
        raise ValueError("Nonfinite pack scores")
    if set(pack_index.tolist()) != set(range(len(true_max))):
        raise ValueError("Every frozen pack must be represented exactly by valid indexes")
    positive = np.maximum(true_max - wrong_max, 0)
    inherited = positive[pack_index]
    primary = bucket_prior(bucket, inherited)
    matched = np.zeros(len(domain))
    for source in np.unique(domain):
        mask = domain == source
        matched[mask] = primary[mask].sum() / mask.sum()
    permutation = np.random.default_rng(permutation_seed).permutation(len(positive))
    shuffled = positive[permutation][pack_index]
    weights = {"contrastive_zipmix": primary, "source_matched": matched,
               "contrastive_shuffled": bucket_prior(bucket, shuffled)}
    status = {m: "available" if weights[m].sum() > 0 else "unavailable_empty_support" for m in METHODS}
    for method in METHODS:
        p = weights[method]
        expected = 1 if status[method] == "available" else 0
        if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), expected):
            raise ValueError("Invalid probability vector")
    return weights, status, positive, permutation, inherited, shuffled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-data', type=Path, default=Path(__file__).parents[2] / '05_validation_guided_sft/expt_v2/data')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Output must be new and empty; do not overwrite a frozen condition')
    parent = json.loads((args.parent_data / 'manifest.json').read_text())
    parent_id = parent.pop('manifest_id')
    if parent_id != PARENT_MANIFEST_ID or stable_hash(parent) != parent_id:
        raise ValueError('Parent manifest differs from the admitted fixed-byte pool')
    parent['manifest_id'] = parent_id
    if parent['model'] != MODEL or parent['model_revision'] != MODEL_REVISION:
        raise ValueError('Parent model differs')
    differences = {k for k in set(CONFIG) | set(parent['config']) if CONFIG.get(k) != parent['config'].get(k)}
    if differences != {'checkpoint_every', 'selector_permutation_seed'}:
        raise ValueError('Unexpected change to inherited training settings')
    inherited_files = ['train.npz', 'target.npz', 'retention.npz', 'pack_scores.npz']
    inherited_files += sorted(k for k in parent['files'] if k.startswith('tokenizer/'))
    # Verify only the inputs needed here, without loading the held-out arrays.
    read_files = ['selection.npz', *inherited_files]
    for name in read_files:
        if file_hash(args.parent_data / name) != parent['files'][name]:
            raise ValueError(f'Frozen parent input changed: {name}')
    with np.load(args.parent_data / 'selection.npz', allow_pickle=False) as a:
        domain, bucket, pack_index = a['domain'], a['bucket'], a['pack_index']
        inherited_true, inherited_wrong = a['score'], a['wrong_score']
    with np.load(args.parent_data / 'pack_scores.npz', allow_pickle=False) as a:
        true_max, wrong_max = a['maximum'], a['wrong_maximum']
        if not (a['width'] == 4096).all() or not np.array_equal(a['bucket'][pack_index], bucket):
            raise ValueError('Parent score units or bin assignments changed')
    if not np.array_equal(true_max[pack_index], inherited_true) or not np.array_equal(wrong_max[pack_index], inherited_wrong):
        raise ValueError('Pack scores do not reproduce inherited example scores')
    with np.load(args.parent_data / 'train.npz', allow_pickle=False) as a:
        if not np.array_equal(a['domain'], np.asarray(parent['domains'])[domain]):
            raise ValueError('Source/example order differs')
        lengths = a['lengths']
    weights, status, positive, permutation, inherited, shuffled = selectors(
        domain, bucket, pack_index, true_max, wrong_max, CONFIG['selector_permutation_seed'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name in inherited_files:
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.parent_data / name, path)
        if file_hash(path) != parent['files'][name]:
            raise ValueError('Snapshot copy changed bytes')
    np.savez_compressed(args.output / 'selection.npz', domain=domain, bucket=bucket,
                        pack_index=pack_index, positive_pack_score=positive,
                        pack_permutation=permutation, contrast_score=inherited,
                        shuffled_contrast_score=shuffled, **weights)
    files = {name: file_hash(args.output / name) for name in [*inherited_files, 'selection.npz']}
    mixtures = {}
    for method, p in weights.items():
        mixtures[method] = {'nonzero_examples': int((p > 0).sum()),
            'effective_examples': float(1 / np.square(p).sum()) if p.sum() else None,
            'source_mass': {name: float(p[domain == i].sum()) for i, name in enumerate(parent['domains'])},
            'bucket_mass': np.bincount(bucket, weights=p, minlength=CONFIG['compression_bins']).tolist(),
            'expected_input_tokens_per_example': float(p @ lengths)}
    source = Path(__file__).parent
    manifest = {'schema': 3, 'version': 'contrastive_v1', 'experiment': '06_contrastive_validation_zipmix',
        'created_at': now(), 'prospective_design': {'experiment05_benchmark_metrics_inspected': False,
            'basis': 'training/development finite-pool scores and completed Experiment04 diagnostics only',
            'timestamp_meaning': 'New selector vectors and protocol frozen before this preparation read any Experiment05 benchmark outcomes'},
        'parent': {'experiment': '05_validation_guided_sft/expt_v2', 'manifest_id': parent_id,
                   'public_manifest': '../../05_validation_guided_sft/expt_v2/data_manifest.json',
                   'reused_data_files': inherited_files,
                   'input_sha256': {name: parent['files'][name] for name in read_files},
                   'storage': 'byte-identical local snapshot of necessary parent arrays/tokenizer; no raw question text copied'},
        'model': MODEL, 'model_revision': MODEL_REVISION, 'config': CONFIG,
        'methods': list(METHODS), 'seeds': list(SEEDS), 'expected_training_cells': len(METHODS) * len(SEEDS),
        'available_training_cells': sum(v == 'available' for v in status.values()) * len(SEEDS),
        'method_status': status, 'required_base_evaluation': True,
        'domains': parent['domains'], 'counts': parent['counts'],
        'pad_token_id': parent['pad_token_id'], 'label_tokens': parent['label_tokens'],
        'source_revisions': parent['source_revisions'], 'bucket_boundaries': parent['bucket_boundaries'],
        'selector': {'primary': 'per-pack max(true_max-wrong_max,0), inherited by each example, summed per bin, uniform within bin',
            'source_matched': 'same exact primary source masses, uniform within each source',
            'shuffled': 'one fixed permutation of positive pack contrast scores before inheritance and bucket aggregation',
            'permutation_seed': CONFIG['selector_permutation_seed'], 'zero_mass': 'explicit unavailable; no substituted prior',
            'positive_packs': int((positive > 0).sum()), 'positive_examples': int((inherited > 0).sum())},
        'mixtures': mixtures,
        'total_variation': {f'{a}_vs_{b}': float(np.abs(weights[a] - weights[b]).sum() / 2)
                            for i, a in enumerate(METHODS) for b in METHODS[i+1:]},
        'budget': {'maximum_device_hours': 2, 'training_cells': 9, 'supervised_tokens_per_cell': 2048,
                   'checkpoint_interval_operational_change': '64 steps instead of parent16; learning and optimizer settings unchanged; wall-time comparisons confounded by less checkpoint I/O'},
        'leakage_provenance': {'parent_13gram_postfilter_hits': parent['postfilter_train_protected_13gram_hits'],
            'parent_exact_postfilter_hits': parent['postfilter_train_protected_exact_hits'],
            'new_examples': 0, 'new_scoring_views': 0, 'heldout_arrays_loaded_by_preparation': False},
        'preparation_code': {name: file_hash(source / name) for name in ('common.py', 'prepare_selection.py')},
        'reused_trainer_sha256': file_hash(source / 'train_sft.py'), 'files': files,
        'limitations': ['exploratory three-seed follow-up and unadjusted descriptive intervals',
            'same public benchmark pools as parent; no new independent held-out replication',
            'input-token compute varies at fixed supervised answer-token budget',
            'source redundancy, pack density, overflow inheritance and wrong-target choice remain confounders',
            'cross-experiment comparisons deferred until both full matrices complete and labeled exploratory']}
    manifest['manifest_id'] = stable_hash(manifest)
    atomic_json(args.output / 'manifest.json', manifest)
    atomic_json(source / 'data_manifest.json', manifest)
    print(json.dumps({'manifest_id': manifest['manifest_id'], 'available_training_cells': manifest['available_training_cells'],
                     'source_mass': {m: v['source_mass'] for m, v in mixtures.items()},
                     'total_variation': manifest['total_variation']}, indent=2))


if __name__ == '__main__':
    main()
