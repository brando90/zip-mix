from pathlib import Path
import json

import numpy as np
import pytest

from common import CONFIG, METHODS, SEEDS, file_hash, stable_hash
from prepare_selection import PARENT_MANIFEST_ID, bucket_prior, selectors


def fixture():
    return (np.array([0, 1, 0, 1, 0, 1]), np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 0, 1, 1, 2, 3]), np.array([.9, .4, .2, .8]),
            np.array([.2, .5, .1, .1]))


def test_clip_before_bucket_aggregation():
    domain = np.array([0, 1, 0])
    bucket = np.array([0, 0, 1])
    weights, status, *_ = selectors(domain, bucket, np.arange(3),
                                    np.array([0., 3., 2.]), np.array([4., 1., 1.]), 1606)
    assert status['contrastive_zipmix'] == 'available'
    np.testing.assert_allclose(weights['contrastive_zipmix'], [1/3, 1/3, 1/3])


def test_source_matching_preserves_every_source_mass():
    domain, bucket, *rest = fixture()
    weights, *_ = selectors(domain, bucket, *rest, 1606)
    for source in np.unique(domain):
        rows = domain == source
        assert weights['contrastive_zipmix'][rows].sum() == pytest.approx(weights['source_matched'][rows].sum(), abs=1e-15)
        assert np.ptp(weights['source_matched'][rows]) == 0
    for b in np.unique(bucket):
        assert np.ptp(weights['contrastive_zipmix'][bucket == b]) == 0


def test_shuffle_is_one_deterministic_pack_permutation():
    args = fixture()
    first = selectors(*args, 1606)
    second = selectors(*args, 1606)
    _, _, positive, permutation, inherited, shuffled = first
    assert sorted(permutation.tolist()) == list(range(len(positive)))
    np.testing.assert_array_equal(inherited, positive[args[2]])
    np.testing.assert_array_equal(shuffled, positive[permutation][args[2]])
    assert shuffled[0] == shuffled[1]
    for name in METHODS:
        np.testing.assert_array_equal(first[0][name], second[0][name])


def test_zero_contrast_is_explicitly_unavailable():
    domain, bucket, pack, true, wrong = fixture()
    weights, status, *_ = selectors(domain, bucket, pack, wrong, wrong, 1606)
    assert set(status.values()) == {'unavailable_empty_support'}
    assert all(np.count_nonzero(weights[m]) == 0 for m in METHODS)


def test_example_count_not_pack_count_weights_bins():
    weights, *_ = selectors(np.array([0, 0, 1]), np.array([0, 0, 1]),
                            np.array([0, 0, 1]), np.ones(2), np.zeros(2), 1606)
    np.testing.assert_allclose(weights['contrastive_zipmix'], np.ones(3)/3)


@pytest.mark.parametrize('bad', [np.array([np.nan, 1]), np.array([-1., 1]), np.array([np.inf, 1])])
def test_bad_bucket_mass_rejected(bad):
    with pytest.raises(ValueError):
        bucket_prior(np.array([0, 1]), bad)


def test_invalid_pack_mapping_rejected():
    args = list(fixture())
    args[2] = np.array([0, 0, 1, 1, 2, 7])
    with pytest.raises(ValueError):
        selectors(*args, 1606)


def test_frozen_parent_training_settings_and_reviewed_trainer():
    source = Path(__file__).parent
    parent = source.parents[1] / '05_validation_guided_sft/expt_v2'
    manifest = json.loads((parent / 'data_manifest.json').read_text())
    assert manifest['manifest_id'] == PARENT_MANIFEST_ID
    assert len(METHODS)*len(SEEDS) == 9
    changed = {k for k in set(CONFIG) | set(manifest['config']) if CONFIG.get(k) != manifest['config'].get(k)}
    assert changed == {'checkpoint_every', 'selector_permutation_seed'}
    assert CONFIG['checkpoint_every'] == 64
    assert CONFIG['steps']*CONFIG['batch_size'] == 2048
    assert file_hash(source / 'train_sft.py') == file_hash(parent / 'train_sft.py')


def test_local_frozen_artifacts_when_present():
    source = Path(__file__).parent
    if not (source / 'data/manifest.json').exists():
        pytest.skip('Private frozen input snapshot is not included in public source')
    manifest = json.loads((source / 'data/manifest.json').read_text())
    public = json.loads((source / 'data_manifest.json').read_text())
    assert manifest == public
    identity = manifest.pop('manifest_id')
    assert stable_hash(manifest) == identity
    assert manifest['expected_training_cells'] == manifest['available_training_cells'] == 9
    for name, digest in manifest['files'].items():
        assert file_hash(source / 'data' / name) == digest
    for name, digest in manifest['preparation_code'].items():
        assert file_hash(source / name) == digest
    with np.load(source / 'data/selection.npz', allow_pickle=False) as selection, np.load(source / 'data/pack_scores.npz', allow_pickle=False) as packs:
        weights, *_ = selectors(selection['domain'], selection['bucket'], selection['pack_index'],
                               packs['maximum'], packs['wrong_maximum'], CONFIG['selector_permutation_seed'])
        for name in METHODS:
            np.testing.assert_array_equal(weights[name], selection[name])
    assert manifest['prospective_design']['experiment05_benchmark_metrics_inspected'] is False
