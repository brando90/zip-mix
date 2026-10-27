"""Deterministic CPU checks of scientific behavior for Experiment 04 v2 on synthetic arrays only."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from train import (ARMS, CausalLM, Config, causal_token_nll, clipped_excess,
                   doge_update, exponentiated_update, failure_kind, grouped_probabilities,
                   load_pt, sampling_probabilities, train_stage)
from analyze import holm_adjust, mean_interval, paired_summary


def tiny_config(**kwargs):
    settings = dict(vocab_size=11, seq_len=4, d_model=8, n_layers=1, n_heads=2,
                    d_ff=16, batch_size=2, final_steps=2, proxy_steps=2,
                    reference_steps=2, proxy_per_domain_batch=1, dev_batch_size=2,
                    warmup_steps=0, checkpoint_every=1, eval_batch_size=2,
                    device="cpu", precision="fp32", num_threads=1, log_every=1)
    settings.update(kwargs)
    return Config(**settings)


def synthetic_data():
    rng = np.random.default_rng(17)
    return dict(tokens=rng.integers(0, 11, (12, 5), dtype=np.int32),
                domain=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32),
                bucket=np.array([0, 0, 0, 4, 4, 4, 1, 1, 1, 4, 4, 4], dtype=np.int32),
                score=np.array([.1, .2, .3, .4, .5, .6, .9, .95, 1.0, .7, .8, .85], dtype=np.float32),
                mean_score=np.array([.8, .7, .6, .5, .4, .3, .9, .9, .9, .2, .1, .05], dtype=np.float32),
                cr=np.array([.5, .6, .65, .7, .75, .8, .55, .58, .59, .85, .95, 1.2], dtype=np.float32))


@pytest.mark.parametrize("arm", [a for a in ARMS if a not in ("doremi", "doge")])
def test_normalized_static_methods(arm):
    p = sampling_probabilities(arm, synthetic_data(), tiny_config(), 3)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(p >= 0.0)
    assert np.isfinite(p).all()


def test_domain_matched_matches_zipmix_domain_bin_domain_marginals_exactly():
    data, cfg = synthetic_data(), tiny_config()
    p_db = sampling_probabilities("zipmix_domain_bin", data, cfg, 0)
    p_dm = sampling_probabilities("domain_matched_zipmix", data, cfg, 0)
    for d in np.unique(data["domain"]):
        mask = data["domain"] == d
        assert np.isclose(p_db[mask].sum(), p_dm[mask].sum(), atol=1e-12)
        # Uniform within domain for domain_matched_zipmix
        assert np.allclose(p_dm[mask], p_dm[mask][0])
    # Domain 0 has two buckets (0 and 4) with different mean scores, so p_db is not uniform within domain 0
    assert not np.allclose(p_db[data["domain"] == 0], p_dm[data["domain"] == 0])


def test_within_domain_conditional_preserves_joint_domain_bin_mass_and_bounds_ratio():
    data, cfg = synthetic_data(), tiny_config()
    p_cond = sampling_probabilities("within_domain_conditional", data, cfg, 0)
    for d in np.unique(data["domain"]):
        for b in np.unique(data["bucket"]):
            mask = (data["domain"] == d) & (data["bucket"] == b)
            if mask.any():
                assert np.isclose(p_cond[mask].sum(), mask.mean(), atol=1e-12)
                assert p_cond[mask].max() / p_cond[mask].min() <= 4.0 + 1e-12
                # Higher score within cell gets strictly higher weight
                assert p_cond[mask][-1] > p_cond[mask][0]


def test_shuffled_domain_bin_permutes_cells_deterministically():
    data, cfg = synthetic_data(), tiny_config()
    p0 = sampling_probabilities("shuffled_domain_bin", data, cfg, 0)
    p0_again = sampling_probabilities("shuffled_domain_bin", data, cfg, 0)
    p1 = sampling_probabilities("shuffled_domain_bin", data, cfg, 1)
    assert np.array_equal(p0, p0_again)
    assert not np.allclose(p0, p1)


def test_six_seed_sign_test_and_holm_adjustment():
    cand = {s: 1.0 for s in range(6)}
    base = {s: 2.0 for s in range(6)}
    res = paired_summary(cand, base)
    assert res["n"] == 6
    assert res["p_value"] == pytest.approx(2 * (0.5 ** 6))
    assert res["p_value"] == pytest.approx(0.03125)
    sub = paired_summary(cand, base, seed_filter={0, 1, 2})
    assert sub["n"] == 3
    assert sub["p_value"] == pytest.approx(0.25)
    adj = holm_adjust({"a": 0.03125, "b": 0.25})
    assert adj["a"] == pytest.approx(0.0625)
    assert adj["b"] == pytest.approx(0.25)


def test_dual_window_evaluation_cli_and_analysis(tmp_path):
    data_dir, output_dir = tmp_path / "data", tmp_path / "results"
    data_dir.mkdir()
    data = synthetic_data()
    np.savez(data_dir / "train.npz", **data)
    rng = np.random.default_rng(37)
    np.savez(data_dir / "dev.npz", tokens=rng.integers(0, 11, (4, 5), dtype=np.int32), domain=np.zeros(4, dtype=np.int32))
    np.savez(data_dir / "eval.npz", target_tokens=rng.integers(0, 11, (4, 5), dtype=np.int32),
             target_domain=np.array([0, 0, 2, 2], dtype=np.int32),
             broad_tokens=rng.integers(0, 11, (4, 5), dtype=np.int32),
             broad_domain=np.array([0, 0, 2, 2], dtype=np.int32))
    np.savez(data_dir / "eval_window2.npz", target_tokens=rng.integers(0, 11, (3, 5), dtype=np.int32),
             target_domain=np.array([0, 2, 2], dtype=np.int32),
             broad_tokens=rng.integers(0, 11, (3, 5), dtype=np.int32),
             broad_domain=np.array([0, 1, 2], dtype=np.int32))
    cfg = tiny_config(
        arms=["token_proportional", "zipmix_domain_bin", "domain_matched_zipmix", "within_domain_conditional", "doremi", "doge"],
        seeds=[0], final_steps=1, proxy_steps=1, reference_steps=1
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(cfg)))
    script_dir = Path(__file__).parent
    command = [sys.executable, str(script_dir / "train.py"), "--data-dir", str(data_dir),
               "--output-dir", str(output_dir), "--config", str(config_path)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    from analyze import analyze
    analysis = analyze(output_dir)
    assert analysis["full_clean_matrix"]
    assert "fresh_target" in analysis["available_splits"]
    assert "fresh_broad" in analysis["available_splits"]
    assert analysis["primary_arm"] == "zipmix_domain_bin"
    assert "domain_matched_zipmix" in analysis["primary_minus_baseline"]
    assert "fresh_target" in analysis["within_domain_conditional_minus_token_proportional"]
