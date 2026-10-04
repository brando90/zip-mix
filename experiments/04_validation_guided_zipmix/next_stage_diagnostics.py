#!/usr/bin/env python3
"""Reproduce proposed-design diagnostics from frozen training/dev metadata only.

Run from any directory; override --data-dir for an existing private snapshot.
Requires NumPy and SciPy. Does not load held-out arrays or outcome receipts,
write probability vectors, modify frozen inputs, or execute any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import rankdata

MANIFEST_ID = '0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44'
TOLERANCE = 1e-12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def synthetic_checks():
    """Persist the original 100-draw checks, counting every scalar identity."""
    rng = np.random.default_rng(7)
    counts = {'draws': 100, 'bin_convex_bound': 0, 'fourfold_tilt_envelope': 0,
              'fixed_joint_mass': 0}
    for _ in range(counts['draws']):
        q = rng.random(20); q /= q.sum()
        bucket, source = np.arange(20) % 4, np.arange(20) % 3
        event = rng.integers(0, 2, 20).astype(bool)
        pi = rng.random(4); pi /= pi.sum()
        p = np.array([pi[b] * q[i] / q[bucket == b].sum() for i, b in enumerate(bucket)])
        fraction = [q[event & (bucket == b)].sum() / q[bucket == b].sum() for b in range(4)]
        require(p[event].sum() <= max(fraction) + TOLERANCE, 'Convex bound failed')
        counts['bin_convex_bound'] += 1
        weight = np.where(event, 4., 1.)
        tilted = np.array([pi[b] * q[i] * weight[i] / (q[bucket == b] * weight[bucket == b]).sum()
                           for i, b in enumerate(bucket)])
        envelope = sum(pi[b] * 4 * fraction[b] / (1 + 3 * fraction[b]) for b in range(4))
        require(abs(tilted[event].sum() - envelope) < TOLERANCE, 'Tilt envelope failed')
        counts['fourfold_tilt_envelope'] += 1
        for s, b in sorted(set(zip(source.tolist(), bucket.tolist()))):
            rows = (source == s) & (bucket == b)
            joint = q[rows].sum()
            conditional = joint * q[rows] * weight[rows] / (q[rows] * weight[rows]).sum()
            require(abs(conditional.sum() - joint) < TOLERANCE, 'Joint mass failed')
            counts['fixed_joint_mass'] += 1
    counts['scalar_identity_checks'] = sum(v for k, v in counts.items() if k != 'draws')
    return {'seed': 7, 'units_per_draw': 20, 'bins_per_draw': 4, 'sources_per_draw': 3,
            'tolerance': TOLERANCE, 'counts': counts, 'all_passed': True}


def analyze(data_dir, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.pop('manifest_id')
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    require(identity == MANIFEST_ID and hashlib.sha256(canonical).hexdigest() == identity,
            'Unexpected parent manifest')
    filenames = ('selection.npz', 'pack_scores.npz', 'train.npz')
    hashes = {name: digest(data_dir / name) for name in filenames}
    require(all(hashes[n] == manifest['files'][n] for n in filenames), 'Input hash mismatch')
    with np.load(data_dir / 'selection.npz', allow_pickle=False) as arrays:
        source, bucket, pack = arrays['domain'], arrays['bucket'], arrays['pack_index']
    with np.load(data_dir / 'pack_scores.npz', allow_pickle=False) as arrays:
        contrast = arrays['maximum'] - arrays['wrong_maximum']
        require((arrays['width'] == 4096).all(), 'Scoring views are not 4096 bytes')
        require(np.array_equal(arrays['bucket'][pack], bucket), 'Pack/bin mapping changed')
    with np.load(data_dir / 'train.npz', allow_pickle=False) as arrays:
        lengths = arrays['lengths']
        require(np.array_equal(arrays['domain'], np.asarray(manifest['domains'])[source]), 'Question order changed')
    require(len(pack) == manifest['counts']['train'] and len(contrast) == manifest['counts']['training_packs'], 'Pool size changed')
    require(np.isfinite(contrast).all() and set(pack.tolist()) == set(range(len(contrast))), 'Invalid scores or pack indexes')
    pack_source = np.array([np.unique(source[pack == i]).item() for i in range(len(contrast))])
    pack_bucket = np.array([np.unique(bucket[pack == i]).item() for i in range(len(contrast))])
    pack_size = np.bincount(pack)
    rank, shuffled = np.zeros(len(contrast)), np.zeros(len(contrast))
    rng = np.random.default_rng(1707)
    strata, masks = [], []
    checks = {'normalization': 0, 'strict_positivity': 0, 'joint_mass': 0,
              'source_mass': 0, 'bucket_mass': 0, 'pack_permutation_multiset': 0,
              'bounded_within_stratum_weight_ratio': 0}
    for s, b in sorted(set(zip(pack_source.tolist(), pack_bucket.tolist()))):
        indexes = np.flatnonzero((pack_source == s) & (pack_bucket == b))
        values = (rankdata(contrast[indexes], method='average') - .5) / len(indexes) - .5
        rank[indexes], shuffled[indexes] = values, rng.permutation(values)
        require(np.array_equal(np.sort(values), np.sort(shuffled[indexes])), 'Permutation changed pack ranks')
        checks['pack_permutation_multiset'] += 1
        for scores in (rank, shuffled):
            weights = np.exp(np.log(4) * scores[indexes])
            require(weights.max() / weights.min() <= 4 + TOLERANCE, 'Weight ratio exceeds four')
            checks['bounded_within_stratum_weight_ratio'] += 1
        rows = (source == s) & (bucket == b)
        masks.append(rows)
        strata.append({'source': manifest['domains'][s], 'bin': int(b), 'packs': len(indexes),
                       'questions': int(rows.sum()), 'distinct_signed_scores': len(np.unique(contrast[indexes])),
                       'distinct_positive_clipped_scores': len(np.unique(np.maximum(contrast[indexes], 0))),
                       'pack_size_min': int(pack_size[indexes].min()), 'pack_size_max': int(pack_size[indexes].max()),
                       'fixed_joint_mass': float(rows.mean())})

    def conditional(scores):
        weights = np.exp(np.log(4) * scores[pack])
        p = np.zeros(len(pack))
        for rows in masks:
            p[rows] = rows.mean() * weights[rows] / weights[rows].sum()
        return p

    probabilities = {'aligned': conditional(rank), 'uniform': np.full(len(pack), 1 / len(pack)),
                     'shuffled': conditional(shuffled)}
    metrics = {}
    max_joint_error = 0.
    for method, p in probabilities.items():
        require(np.isfinite(p).all() and (p > 0).all(), 'Invalid probabilities')
        checks['strict_positivity'] += 1
        require(abs(p.sum() - 1) < TOLERANCE, 'Normalization failed')
        checks['normalization'] += 1
        for row, mask in zip(strata, masks):
            error = abs(p[mask].sum() - mask.mean())
            require(error < TOLERANCE, 'Joint exposure changed')
            checks['joint_mass'] += 1
            row.setdefault('actual_joint_mass', {})[method] = float(p[mask].sum())
            max_joint_error = max(max_joint_error, float(error))
        for labels, key in ((source, 'source_mass'), (bucket, 'bucket_mass')):
            for label in np.unique(labels):
                mask = labels == label
                require(abs(p[mask].sum() - mask.mean()) < TOLERANCE, 'Marginal exposure changed')
                checks[key] += 1
        metrics[method] = {'effective_sample_size': float(1 / (p @ p)),
            'effective_sample_fraction': float(1 / (p @ p) / len(p)),
            'expected_input_tokens_per_question': float(p @ lengths),
            'nonzero_questions': int((p > 0).sum()),
            'source_mass': {name: float(p[source == i].sum()) for i, name in enumerate(manifest['domains'])}}
    distances = {f'{a}_vs_{b}': float(np.abs(probabilities[a] - probabilities[b]).sum() / 2)
                 for a, b in (('aligned', 'uniform'), ('aligned', 'shuffled'), ('uniform', 'shuffled'))}
    return {'schema': 1, 'kind': 'deterministic proposed-design diagnostics; not an executed training condition',
        'parent_manifest_id': identity, 'input_sha256': hashes, 'script_sha256': digest(Path(__file__)),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'input_fields_read': {'selection.npz': ['domain', 'bucket', 'pack_index'],
            'pack_scores.npz': ['maximum', 'wrong_maximum', 'bucket', 'width'],
            'train.npz': ['lengths', 'domain']},
        'information_boundary': {'heldout_arrays_read': False, 'experiment05_or_06_outcomes_read': False,
                                  'model_calls': 0, 'raw_arrays_exported': False, 'frozen_inputs_modified': False},
        'recipe': {'contrast': 'signed maximum(true development alignment)-maximum(wrong development alignment); no clipping',
            'rank': 'average rank R within each source x compression stratum over its m packs; z=(R-0.5)/m-0.5',
            'weight': 'exp(log(4)*z), inherited to all pack members',
            'normalization': 'p_i=(n_source,bin/N)*w_i/sum_{j in source,bin}(w_j)',
            'shuffle': 'permute centered pack ranks within each source x bin stratum; sorted source/bin order, frozen pack-index order',
            'permutation_seed': 1707, 'constant_or_singleton_stratum': 'zero centered ranks, therefore uniform',
            'permutation_caveat': 'pack-rank multiset preserved; question-weight multiset need not be preserved with unequal pack sizes'},
        'counts': {'questions': len(pack), 'packs': len(contrast), 'nonempty_strata': len(strata),
            'constant_signed_strata': sum(r['distinct_signed_scores'] == 1 for r in strata),
            'constant_positive_clipped_strata': sum(r['distinct_positive_clipped_scores'] == 1 for r in strata),
            'minimum_packs_per_stratum': min(r['packs'] for r in strata)},
        'strata': strata, 'metrics': metrics, 'total_variation': distances,
        'maximum_joint_mass_absolute_error': max_joint_error,
        'frozen_pool_probability_checks': {'counts': checks, 'assertion_checks': sum(checks.values()),
            'finite_probability_entries_checked': 3 * len(pack),
            'strictly_positive_probability_entries_checked': 3 * len(pack),
            'pack_rank_permutation_entries_checked': len(contrast),
            'counting_unit': 'one assertion per vector or stratum as named; tested element counts are separate',
            'tolerance': TOLERANCE, 'all_passed': True},
        'synthetic_probability_checks': synthetic_checks(),
        'interpretation': 'Exact finite-pool diagnostics. No performance estimate, confidence interval or p-value applies.'}


def main():
    home = Path(__file__).resolve().parent
    parent = home.parent / '05_validation_guided_sft/expt_v2'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=parent / 'data')
    parser.add_argument('--manifest', type=Path, default=parent / 'data_manifest.json')
    parser.add_argument('--output', type=Path, default=home / 'expt_v1/results/next_stage_diagnostics.json')
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.data_dir.resolve())
            and args.output.resolve() != args.manifest.resolve(), 'Output cannot overwrite frozen inputs')
    receipt = analyze(args.data_dir, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'questions': receipt['counts']['questions'], 'strata': receipt['counts']['nonempty_strata'],
                      'frozen_pool_checks': receipt['frozen_pool_probability_checks']['assertion_checks'],
                      'synthetic_checks': receipt['synthetic_probability_checks']['counts']['scalar_identity_checks'],
                      'total_variation': receipt['total_variation']}))


if __name__ == '__main__':
    main()
