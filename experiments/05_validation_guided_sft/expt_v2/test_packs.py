"""Checks that the prospective experiment changes its intended scoring unit."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from common import CONFIG
from prepare_sft import alignment, bucket_weights, compute_mixtures, make_packs


def test_packing_uses_only_real_prefixes_and_partitions_records():
    rows = [{"id": f"r{i:02}", "content": chr(65 + i) * 20} for i in range(8)]
    packs, tail = make_packs(rows, "test", width=64)
    assert len(packs) == 2 and tail == []
    assert all(len(p["view"]) == 64 for p in packs)
    members = [r["id"] for p in packs for r in p["records"]]
    assert members == [r["id"] for r in rows]
    for pack in packs:
        assert pack["view"] == b"".join(r["content"].encode() + b"\n" for r in pack["records"])[:64]
        assert pack["full_bytes"] == 84


def test_short_tail_is_dropped_not_padded_or_repeated():
    rows = [{"id": str(i), "content": "abcdefghij"} for i in range(7)]
    packs, tail = make_packs(rows, "test", width=64)
    assert len(packs) == 1 and len(tail) == 1
    assert len(packs[0]["records"]) == 6
    assert tail[0]["id"] == "6"
    assert len(packs[0]["view"]) == 64


def test_pack_identity_and_order_are_reproducible():
    rows = [{"id": str(i), "content": "question" * 10} for i in range(5)]
    a, tail_a = make_packs(rows, "test", width=100)
    b, tail_b = make_packs(list(reversed(rows)), "test", width=100)
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert [p["view_sha256"] for p in a] == [p["view_sha256"] for p in b]
    assert tail_a == tail_b


def test_alignment_rejects_unequal_byte_width():
    width = CONFIG["score_byte_window"]
    with pytest.raises(ValueError, match="identical byte length"):
        alignment(b"x" * width, [b"y" * (width - 1)])
    maximum, mean = alignment(b"science?" * (width // 8), [b"science?" * (width // 8)])
    assert np.isfinite(maximum) and maximum == mean


def inputs():
    # Two packs, each with two answer-token examples.
    return dict(domain=np.array([0, 0, 1, 1]), bucket=np.array([0, 0, 1, 1]),
                cr=np.array([.7, .7, .75, .75]), score_max=np.array([3., 3., 1., 1.]),
                score_mean=np.array([.3, .3, .1, .1]), wrong_max=np.array([1., 1., 3., 3.]),
                pack_index=np.array([0, 0, 1, 1]), ids=["a", "b", "c", "d"], pack_scores=np.array([3., 1.]))


def test_true_and_wrong_targets_can_change_prior_with_identical_bins():
    weights, status = compute_mixtures(**inputs())
    np.testing.assert_allclose(weights["zipmix_static"], [.375, .375, .125, .125])
    np.testing.assert_allclose(weights["wrongtarget_zipmix"], [.125, .125, .375, .375])
    assert all(s == "available" for s in status.values())
    assert all(np.isclose(w.sum(), 1) for w in weights.values())


def test_empty_compel_is_a_declared_unavailable_method():
    values = inputs()
    values["cr"] = np.array([.4, .4, .5, .5])
    weights, status = compute_mixtures(**values)
    assert status["compel_filter"] == "unavailable_empty_support"
    assert np.array_equal(weights["compel_filter"], np.zeros(4))
    assert len(weights) == 7


def test_bucket_population_is_supervised_example_mass():
    bucket = np.array([0, 0, 0, 1])
    np.testing.assert_allclose(bucket_weights(bucket, np.ones(4)), np.ones(4) / 4)
    result = bucket_weights(bucket, np.array([2., 2., 2., 1.]))
    np.testing.assert_allclose(result, [2 / 7, 2 / 7, 2 / 7, 1 / 7])
