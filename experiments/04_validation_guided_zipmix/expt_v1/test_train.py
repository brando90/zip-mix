"""Deterministic CPU checks of scientific behavior on synthetic arrays only."""
import json
from pathlib import Path
from dataclasses import asdict
import subprocess
import sys

import numpy as np
import pytest
import torch

from train import (ARMS, CausalLM, Config, causal_token_nll, clipped_excess,
                   doge_update, exponentiated_update, failure_kind, grouped_probabilities,
                   load_pt, sampling_probabilities, train_stage)
from analyze import mean_interval, paired_summary


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
    return dict(tokens=rng.integers(0, 11, (8, 5), dtype=np.int32),
                domain=np.array([0, 0, 0, 0, 0, 0, 2, 2], dtype=np.int32),
                bucket=np.array([0, 0, 0, 4, 4, 4, 4, 4], dtype=np.int32),
                score=np.array([.1, .2, .3, .4, .5, .6, .7, .8], dtype=np.float32),
                mean_score=np.array([.8, .7, .6, .5, .4, .3, .2, .1], dtype=np.float32),
                cr=np.array([.5, .6, .65, .7, .75, .8, .85, 1.2], dtype=np.float32))


@pytest.mark.parametrize("arm", ARMS[:7])
def test_normalized_static_methods(arm):
    p = sampling_probabilities(arm, synthetic_data(), tiny_config(), 3)
    assert np.isclose(p.sum(), 1)
    assert np.all(p >= 0)
    assert np.isfinite(p).all()


def test_empty_bucket_has_no_mass_and_formula_is_exact():
    data, cfg = synthetic_data(), tiny_config()
    p = sampling_probabilities("zipmix_static", data, cfg, 0)
    # Labels 1,2,3 are empty: smoothing must not reserve probability for them.
    mass0 = np.maximum(data["score"][data["bucket"] == 0], 0).sum(dtype=np.float64) + cfg.tau
    mass4 = np.maximum(data["score"][data["bucket"] == 4], 0).sum(dtype=np.float64) + cfg.tau
    assert np.isclose(p[data["bucket"] == 0].sum(), mass0 / (mass0 + mass4))
    assert np.isclose(p.sum(), 1)
    assert p[0] == p[1] == p[2]


def test_proportional_and_uniform_are_distinct():
    data, cfg = synthetic_data(), tiny_config()
    tp = sampling_probabilities("token_proportional", data, cfg, 0)
    uniform = sampling_probabilities("uniform_domain", data, cfg, 0)
    assert np.isclose(tp[data["domain"] == 0].sum(), .75)
    assert np.isclose(uniform[data["domain"] == 0].sum(), .5)


def test_compel_fails_when_empty():
    data = synthetic_data()
    data["cr"][:] = 0.2
    with pytest.raises(ValueError, match="no chunks"):
        sampling_probabilities("compel_filter", data, tiny_config(), 0)


def test_direct_zipfit_uses_mean_not_max():
    data, cfg = synthetic_data(), tiny_config()
    p = sampling_probabilities("direct_zipfit", data, cfg, 0)
    assert p[0] > p[-1]  # max-score order is opposite.
    del data["mean_score"]
    with pytest.raises(ValueError, match="mean_score"):
        sampling_probabilities("direct_zipfit", data, cfg, 0)


def test_shuffled_control_is_fixed_and_breaks_association():
    data, cfg = synthetic_data(), tiny_config()
    p = sampling_probabilities("shuffled_zipmix", data, cfg, 7)
    assert np.array_equal(p, sampling_probabilities("shuffled_zipmix", data, cfg, 7))
    assert not np.allclose(p, sampling_probabilities("zipmix_static", data, cfg, 7))


def test_causal_targets_are_shifted():
    class Oracle(torch.nn.Module):
        def forward(self, x):
            assert x.tolist() == [[0, 1, 2]]
            return torch.nn.functional.one_hot(x + 1, num_classes=4).float() * 20
    nll = causal_token_nll(Oracle(), torch.tensor([[0, 1, 2, 3]]))
    assert nll.shape == (1, 3)
    assert nll.max() < 1e-6


def test_future_tokens_cannot_change_past_logits():
    torch.manual_seed(1)
    model = CausalLM(tiny_config()).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 9, 8]])
    assert torch.allclose(model(a)[:, :2], model(b)[:, :2], atol=1e-7, rtol=0)


def test_excess_clipped_per_token_before_mean():
    loss = torch.tensor([[1.0, 5.0]], requires_grad=True)
    ref = torch.tensor([[4.0, 2.0]])
    excess = clipped_excess(loss, ref)
    assert excess.item() == 1.5
    assert (loss - ref).mean().clamp_min(0).item() == 0
    weights = exponentiated_update(torch.tensor([.5, .5]), torch.tensor([1.5, 0.0]), smoothing=.001)
    assert weights[0] > weights[1]
    assert weights.sum().item() == pytest.approx(1.0)


def test_doge_upweights_alignment_and_downweights_opposition():
    train = torch.tensor([[1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    target = torch.tensor([1.0, 0.0, 1.0])
    weights, scores = doge_update(torch.tensor([1/3] * 3), train, target, lr=.01, mu=.01)
    assert scores[0] > 0 > scores[1]
    assert weights[0] > weights[2] > weights[1]
    assert weights.sum().item() == pytest.approx(1.0)
    assert not target.requires_grad


def test_doge_real_development_gradient_changes_mixture(tmp_path):
    torch.set_num_threads(1)
    cfg = tiny_config(proxy_steps=1, doge_mu=.00001, learning_rate=.01)
    train = synthetic_data()
    train["tokens"][:6] = np.array([1, 1, 1, 1, 1])
    train["tokens"][6:] = np.array([2, 2, 2, 2, 2])
    dev1 = dict(tokens=np.tile([1, 1, 1, 1, 1], (4, 1)).astype(np.int32))
    dev2 = dict(tokens=np.tile([2, 2, 2, 2, 2], (4, 1)).astype(np.int32))
    _, result1 = train_stage(cfg, train, dev1, tmp_path / "a", "proxy", 3, "test", method="doge")
    _, result2 = train_stage(cfg, train, dev2, tmp_path / "b", "proxy", 3, "test", method="doge")
    assert result1["average_domain_weights"][0] > result2["average_domain_weights"][0]
    assert result1["gradient_probe_tokens"] == cfg.dev_batch_size * cfg.seq_len


def test_resume_restores_model_optimizer_rng_and_budget(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = tiny_config(final_steps=3)
    data = synthetic_data()
    p = sampling_probabilities("token_proportional", data, cfg, 0)
    full, full_stats = train_stage(cfg, data, {}, tmp_path / "full", "final", 0, "test", probabilities=p)
    import train as module
    real_atomic = module.atomic_write
    def interrupt_after_first(path, writer):
        real_atomic(path, writer)
        if str(path).endswith("final_checkpoint.pt"):
            raise KeyboardInterrupt("synthetic interruption after a durable checkpoint")
    monkeypatch.setattr(module, "atomic_write", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        train_stage(cfg, data, {}, tmp_path / "resume", "final", 0, "test", probabilities=p)
    monkeypatch.setattr(module, "atomic_write", real_atomic)
    resumed, resumed_stats = train_stage(cfg, data, {}, tmp_path / "resume", "final", 0, "test", probabilities=p)
    for a, b in zip(full.parameters(), resumed.parameters()):
        assert torch.equal(a, b)
    assert resumed_stats["optimizer_tokens"] == full_stats["optimizer_tokens"] == 24
    assert load_pt(tmp_path / "resume" / "final_checkpoint.pt")["step"] == 3


def test_statistics_use_seed_pairs_and_honest_small_n():
    result = paired_summary({0: 1, 1: 2, 2: 3}, {0: 2, 1: 3, 2: 4})
    assert result["n"] == 3
    assert result["mean"] == -1
    assert result["p_value"] == .25
    assert result["ci95"] == [-1, -1]
    assert mean_interval([1, 2])["ci95"] is None
    partial = paired_summary({0: 1, 1: 2, 2: 3}, {1: 1, 3: 1})
    assert partial["n"] == 1
    assert partial["paired_seeds"] == [1]


def test_all_arms_cli_manifest_and_analysis(tmp_path):
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
    cfg = tiny_config(seeds=[0], final_steps=1, proxy_steps=1, reference_steps=1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(cfg)))
    script_dir = Path(__file__).parent
    command = [sys.executable, str(script_dir / "train.py"), "--data-dir", str(data_dir),
               "--output-dir", str(output_dir), "--config", str(config_path)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["expected_cells"] == 9
    assert manifest["counts"]["complete"] == 9
    assert manifest["full_clean_matrix"]
    # Re-running a complete identity does not retrain or rewrite cell results.
    metrics_path = output_dir / "zipmix_static_seed0" / "metrics.json"
    previous = metrics_path.read_bytes()
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert metrics_path.read_bytes() == previous
    from analyze import analyze
    analysis = analyze(output_dir)
    assert analysis["full_clean_matrix"]
    assert analysis["zipmix_minus_baseline"]["doremi"]["target"]["n"] == 1
    assert analysis["zipmix_minus_baseline"]["doge"]["target"]["ci95"] is None
    stages = json.loads((output_dir / "doremi_seed0" / "metrics.json").read_text())["stages"]
    assert set(stages) == {"reference", "proxy", "final"}
    assert stages["reference"]["seen_by_domain"]
    assert stages["proxy"]["forward_only_tokens"] == 8


def test_failure_kind_separates_numerical_from_infrastructure():
    assert failure_kind(FloatingPointError("nonfinite final training loss at step 3")) == "numerical"
    assert failure_kind(RuntimeError("The total norm of order 2.0 for gradients is non-finite")) == "numerical"
    assert failure_kind(KeyboardInterrupt("SIGTERM")) == "interrupted"
    assert failure_kind(OSError("disk quota exceeded")) == "runtime"


def write_cli_inputs(tmp_path, **config):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(41)
    np.savez(data_dir / "train.npz", **synthetic_data())
    np.savez(data_dir / "dev.npz", tokens=rng.integers(0, 11, (4, 5), dtype=np.int32), domain=np.zeros(4, dtype=np.int32))
    np.savez(data_dir / "eval.npz", target_tokens=rng.integers(0, 11, (2, 5), dtype=np.int32),
             target_domain=np.array([0, 2], dtype=np.int32), broad_tokens=rng.integers(0, 11, (2, 5), dtype=np.int32),
             broad_domain=np.array([0, 2], dtype=np.int32))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(tiny_config(**config))))
    return [sys.executable, str(Path(__file__).parent / "train.py"), "--data-dir", str(data_dir),
            "--output-dir", str(tmp_path / "results"), "--config", str(config_path)], data_dir


def test_retry_skips_numerical_failures_but_resumes_runtime_failures(tmp_path):
    command, _ = write_cli_inputs(tmp_path, arms=["token_proportional"], seeds=[0])
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0
    manifest_path = tmp_path / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0].update(status="failed", failure_kind="numerical", error="synthetic")
    manifest_path.write_text(json.dumps(manifest))
    assert subprocess.run(command + ["--retry-failed"], capture_output=True, text=True).returncode == 1
    cell = json.loads(manifest_path.read_text())["cells"][0]
    assert cell["status"] == "failed" and cell["recoveries"] == 0
    manifest["cells"][0]["failure_kind"] = "runtime"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(command + ["--retry-failed"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    cell = json.loads(manifest_path.read_text())["cells"][0]
    assert cell["status"] == "complete" and cell["recoveries"] == 1
    assert cell["recovery_history"][0]["prior_status"] == "failed"


def test_data_must_match_frozen_preparation_manifest(tmp_path):
    command, data_dir = write_cli_inputs(tmp_path, arms=["token_proportional"], seeds=[0])
    (data_dir / "manifest.json").write_text(json.dumps({"files": {"train.npz": "0" * 64}}))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "frozen preparation manifest" in result.stderr
