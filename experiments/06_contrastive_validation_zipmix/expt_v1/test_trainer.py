"""Inherited trainer checks plus the prospective64-step checkpoint recovery."""
import numpy as np
import pytest
import torch
from common import stable_hash
from train_sft import left_padded_batch, last_logits, new_model

def test_left_padding_preserves_actual_final_token_and_positions():
    data = {"lengths": np.array([2, 4]), "input_ids": np.array([[11, 12, 0, 0], [21, 22, 23, 24]])}
    result, tokens = left_padded_batch(data, [0, 1], 99, torch.device("cpu"))
    assert tokens == 6
    assert result["input_ids"].tolist() == [[99, 99, 11, 12], [21, 22, 23, 24]]
    assert result["attention_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert result["position_ids"].tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]


def test_last_logit_interface_uses_one_logit_position():
    class Dummy:
        def forward(self, input_ids, use_cache=False, logits_to_keep=None):
            assert not use_cache and logits_to_keep == 1
            return type("Output", (), {"logits": torch.ones(2, 1, 9)})()
        def __call__(self, **kwargs):
            return self.forward(**kwargs)
    assert last_logits(Dummy(), {"input_ids": torch.ones(2, 4)}).shape == (2, 9)


def test_one_position_cross_entropy_equals_masked_causal_loss():
    torch.manual_seed(4)
    logits = torch.randn(3, 5, 11)
    target = torch.tensor([1, 4, 8])
    labels = torch.full((3, 5), -100)
    labels[:, -1] = target
    masked = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1))
    answer_only = torch.nn.functional.cross_entropy(logits[:, -1], target)
    torch.testing.assert_close(masked, answer_only)


def test_hash_detects_configuration_change():
    assert stable_hash({"steps": 128, "seed": 0}) == stable_hash({"seed": 0, "steps": 128})
    assert stable_hash({"steps": 128}) != stable_hash({"steps": 129})


def test_model_loading_uses_float32_master_parameters(monkeypatch):
    import train_sft
    calls = {}

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
            self.config = type("Config", (), {"use_cache": True})()

    def load(*args, **kwargs):
        calls.update(kwargs)
        return Tiny()

    monkeypatch.setattr(train_sft.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    model = new_model(None, torch.device("cpu"), 0)
    assert calls["torch_dtype"] is torch.float32
    assert all(p.dtype is torch.float32 and p.requires_grad for p in model.parameters())
    assert model.config.use_cache is False


def test_left_padded_last_logits_match_unpadded_qwen2_architecture():
    """Tiny random Qwen2 (same masking/position code path, no weights or network)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         tie_word_embeddings=True)
    model = Qwen2ForCausalLM._from_config(config, attn_implementation="sdpa", torch_dtype=torch.float32).eval()
    data = {"lengths": np.array([3, 9, 6]),
            "input_ids": np.array([[5, 6, 7] + [0] * 6, list(range(10, 19)), [20, 21, 22, 23, 24, 25, 0, 0, 0]])}
    device = torch.device("cpu")
    with torch.inference_mode():
        batched = last_logits(model, left_padded_batch(data, np.arange(3), 63, device)[0])
        single = torch.cat([last_logits(model, left_padded_batch(data, np.array([i]), 63, device)[0]) for i in range(3)])
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-5)


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)
        self.config = type("Config", (), {"use_cache": False})()

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        hidden = self.embed(input_ids) * attention_mask[..., None]
        return type("Output", (), {"logits": self.head(hidden.cumsum(1))})()


def _resume_fixture(monkeypatch):
    import train_sft
    def tiny_model(cache, device, seed):
        torch.manual_seed(seed)
        return _TinyLM()
    monkeypatch.setattr(train_sft, "new_model", tiny_model)
    for name, value in (("reset_peak_memory_stats", lambda: None), ("synchronize", lambda: None),
                        ("max_memory_allocated", lambda: 0), ("empty_cache", lambda: None),
                        ("get_rng_state_all", lambda: []), ("set_rng_state_all", lambda states: None)):
        monkeypatch.setattr(torch.cuda, name, value)
    rng = np.random.default_rng(3)
    lengths = rng.integers(2, 6, 12)
    split = {"ids": np.array([f"x{i}" for i in range(12)]), "lengths": lengths,
             "input_ids": rng.integers(6, 16, (12, 6)), "nchoices": np.full(12, 4),
             "answer": rng.integers(0, 4, 12)}
    split["answer_token"] = split["answer"] + 1
    manifest = {"pad_token_id": 0, "label_tokens": [1, 2, 3, 4, 5], "manifest_id": "synthetic"}
    weights = {"contrastive_zipmix": np.full(12, 1 / 12)}
    return train_sft, split, manifest, weights


def _run_cell(train_sft, output, split, manifest, weights):
    args = type("Args", (), {"output": output, "cache_dir": None, "eval_batch_size": 4})()
    cell = {"id": "contrastive_zipmix__seed0", "method": "contrastive_zipmix", "seed": 0, "status": "pending"}
    ledger = {"run_id": "synthetic", "cells": [cell]}
    return args, cell, ledger


def test_interrupted_cell_resumes_bitwise_and_final_step_resume_completes(tmp_path, monkeypatch):
    train_sft, split, manifest, weights = _resume_fixture(monkeypatch)
    evaluation = {"target": split, "retention": split}
    device = torch.device("cpu")
    args, cell, ledger = _run_cell(train_sft, tmp_path / "full", split, manifest, weights)
    train_sft.train_cell(cell, ledger, tmp_path / "full.json", args, manifest, split, evaluation, weights, device)
    reference = torch.load(tmp_path / "full" / cell["id"] / "model.pt")["state_dict"]
    # Interrupt immediately after the first durable checkpoint, then resume once.
    real_save = train_sft.atomic_torch
    def interrupt(path, value):
        real_save(path, value)
        if path.name == "checkpoint.pt":
            raise KeyboardInterrupt("synthetic interruption")
    monkeypatch.setattr(train_sft, "atomic_torch", interrupt)
    args, cell, ledger = _run_cell(train_sft, tmp_path / "resume", split, manifest, weights)
    with pytest.raises(KeyboardInterrupt):
        train_sft.train_cell(cell, ledger, tmp_path / "resume.json", args, manifest, split, evaluation, weights, device)
    monkeypatch.setattr(train_sft, "atomic_torch", real_save)
    assert cell["status"] == "running"
    train_sft.train_cell(cell, ledger, tmp_path / "resume.json", args, manifest, split, evaluation, weights, device)
    resumed = torch.load(tmp_path / "resume" / cell["id"] / "model.pt")["state_dict"]
    assert all(torch.equal(reference[k], resumed[k]) for k in reference)
    assert cell["status"] == "complete" and cell["resumes"][0]["checkpoint_step"] == 64
    # Crash during evaluation after the final-step checkpoint: resume runs no
    # training step and must still finish complete, not as a failed row.
    real_evaluate = train_sft.evaluate
    def crash(*a, **k):
        raise KeyboardInterrupt("synthetic evaluation interruption")
    monkeypatch.setattr(train_sft, "evaluate", crash)
    args, cell, ledger = _run_cell(train_sft, tmp_path / "final", split, manifest, weights)
    with pytest.raises(KeyboardInterrupt):
        train_sft.train_cell(cell, ledger, tmp_path / "final.json", args, manifest, split, evaluation, weights, device)
    monkeypatch.setattr(train_sft, "evaluate", real_evaluate)
    train_sft.train_cell(cell, ledger, tmp_path / "final.json", args, manifest, split, evaluation, weights, device)
    assert cell["status"] == "complete"
    assert cell["resumes"][0]["checkpoint_step"] == 128 and cell["resumes"][0]["replayed_optimizer_steps"] == 0
    final = torch.load(tmp_path / "final" / cell["id"] / "model.pt")["state_dict"]
    assert all(torch.equal(reference[k], final[k]) for k in reference)
    assert not (tmp_path / "final" / cell["id"] / "checkpoint.pt").exists()
