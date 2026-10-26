"""Deterministic CPU tests for Experiment 07 (no pretrained weights, no network, no GPU).

Run from this directory: python -m pytest -q test_exp07.py
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (CALIBRATION_LRS, CONFIG, TOLERANCE, select_calibrated_lr,
                    within_stratum_conditional_weights)
from train_exp07 import code_identity, frozen_weights, last_logits, left_padded_batch, plan_cells

HOME = Path(__file__).parent
DATA = HOME / "data"
needs_data = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="frozen data not present")


def load(name):
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


# ---- (a) labels and splits -------------------------------------------------

@needs_data
def test_every_split_has_consistent_labels_and_unique_items():
    manifest = json.loads((DATA / "manifest.json").read_text())
    labels = manifest["label_tokens"]
    for name in ("train", "cal_target", "cal_retention", "target", "retention",
                 "test_holdout2", "reused_target", "reused_retention"):
        d = load(name)
        assert len(d["ids"]) == manifest["counts"][name]
        assert len(set(d["ids"].tolist())) == len(d["ids"])
        assert ((0 <= d["answer"]) & (d["answer"] < d["nchoices"])).all()
        assert all(labels[a] == t for a, t in zip(d["answer"].tolist(), d["answer_token"].tolist()))
        beyond = np.arange(d["input_ids"].shape[1])[None, :] >= d["lengths"][:, None]
        assert (d["input_ids"][beyond] == manifest["pad_token_id"]).all()


@needs_data
def test_splits_are_pairwise_disjoint_by_identity_and_question():
    names = ("train", "cal_target", "cal_retention", "target", "retention", "test_holdout2")
    ids = {n: load(n) for n in names}
    for a, b in itertools.combinations(names, 2):
        assert not set(ids[a]["ids"].tolist()) & set(ids[b]["ids"].tolist()), (a, b)
        assert not set(ids[a]["question_ids"].tolist()) & set(ids[b]["question_ids"].tolist()), (a, b)


# ---- (d) arms differ: frozen selectors reproduce and are distinct ----------

@needs_data
def test_frozen_conditional_selectors_reproduce_and_pass_gates():
    s = load("selection"); ps = load("pack_scores")
    out, diag = within_stratum_conditional_weights(
        s["domain"], s["bucket"], s["pack_index"], ps["maximum"], ps["wrong_maximum"],
        permutation_seed=CONFIG["conditional_permutation_seed"])
    for name, p in out.items():
        np.testing.assert_allclose(p, s[name], rtol=0, atol=1e-15)
    assert diag["maximum_joint_mass_absolute_error"] < TOLERANCE
    assert min(diag["effective_sample_fraction"].values()) >= 0.80
    assert diag["tv_aligned_vs_uniform"] >= 0.05 and diag["tv_aligned_vs_shuffled"] >= 0.05


def test_conditional_weights_are_monotone_in_contrast_within_a_stratum():
    # two strata; stratum (0,0) has three packs with distinct contrasts, stratum (1,1) has two packs
    source = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    bucket = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    pack = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    true_max = np.array([0.1, 0.3, 0.2, 0.5, 0.4])
    wrong_max = np.zeros(5)
    out, diag = within_stratum_conditional_weights(source, bucket, pack, true_max, wrong_max)
    w = out["conditional_aligned"]
    assert w[0] < w[4] < w[2]            # contrast ranks 0.1 < 0.2 < 0.3 inside stratum (0,0)
    assert abs(w[:6].sum() - 0.6) < TOLERANCE and abs(w[6:].sum() - 0.4) < TOLERANCE
    assert max(w[:6]) / min(w[:6]) <= 4.0 + TOLERANCE
    np.testing.assert_allclose(out["stratum_uniform"], 0.1)


# ---- Stage 1 selection rule ------------------------------------------------

def synthetic(target_means, retention_means):
    return {name: {"cal_target": {"accuracy": {"mean": t}, "label_nll_mean": 0.5},
                   "cal_retention": {"accuracy": {"mean": r}}}
            for name, t, r in zip(CALIBRATION_LRS, target_means, retention_means)}


def test_calibration_rule_prefers_highest_target_among_retention_feasible():
    base = {"cal_target": {"accuracy": 0.80}, "cal_retention": {"accuracy": 0.52}}
    metrics = synthetic([0.81, 0.805, 0.80, 0.78, 0.73], [0.55, 0.57, 0.60, 0.60, 0.55])
    decision = select_calibrated_lr(metrics, base)
    assert decision["selected_method"] == "lr_1e6" and decision["selection_reason"].startswith("maximized")
    assert decision["calibration_preserves_or_improves_base_target"]


def test_calibration_rule_retention_floor_excludes_degrading_rates_and_falls_back():
    base = {"cal_target": {"accuracy": 0.80}, "cal_retention": {"accuracy": 0.60}}
    metrics = synthetic([0.80, 0.81, 0.82, 0.83, 0.84], [0.60, 0.595, 0.58, 0.55, 0.50])
    decision = select_calibrated_lr(metrics, base)
    assert decision["selected_method"] == "lr_2e6"          # 0.595 >= 0.60 - 0.010 is still feasible
    metrics = synthetic([0.80, 0.81, 0.82, 0.83, 0.84], [0.50, 0.50, 0.50, 0.50, 0.50])
    decision = select_calibrated_lr(metrics, base)
    assert decision["selection_reason"].startswith("fallback")


# ---- trainer plumbing regressions -----------------------------------------

def test_code_identity_records_missing_files_instead_of_crashing(tmp_path):
    (tmp_path / "common.py").write_text("x = 1\n")
    identity = code_identity(tmp_path)
    assert identity["analyze_exp07.py"] == "missing" and identity["train_exp07.py"] == "missing"
    assert len(identity["common.py"]) == 64


def test_plan_cells_without_method_status_table_marks_every_cell_pending():
    cells = plan_cells("diagnostic", ("conditional_aligned", "conditional_shuffled"), (0, 1), 5e-6, {})
    assert [c["id"] for c in cells] == ["conditional_aligned__seed0", "conditional_aligned__seed1",
                                        "conditional_shuffled__seed0", "conditional_shuffled__seed1"]
    assert all(c["status"] == "pending" and c["learning_rate"] == 5e-6 for c in cells)
    cells = plan_cells("calibration", ("lr_2e6",), (0,), None, {})
    assert cells[0]["selector_name"] == "stratum_uniform" and cells[0]["learning_rate"] == 2e-6
    cells = plan_cells("selectors", ("direct_zipfit",), (0,), 2e-6, {"method_status": {"direct_zipfit": "unavailable"}})
    assert cells[0]["status"] == "failed" and cells[0]["failure_kind"] == "scientific_method_unavailable"


def test_left_padding_positions_start_at_zero_on_first_real_token():
    data = {"input_ids": np.array([[5, 6, 7, 0], [8, 0, 0, 0]], dtype=np.int32),
            "lengths": np.array([3, 1], dtype=np.int32)}
    inputs, count = left_padded_batch(data, np.array([0, 1]), pad_token=0, device=torch.device("cpu"))
    assert count == 4
    assert inputs["input_ids"].tolist() == [[5, 6, 7], [0, 0, 8]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert inputs["position_ids"].tolist() == [[0, 1, 2], [0, 0, 0]]


def test_last_logits_equals_final_position_of_full_logits():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
                         tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config).eval()
    data = {"input_ids": np.array([[3, 4, 5, 6], [7, 8, 0, 0]], dtype=np.int32),
            "lengths": np.array([4, 2], dtype=np.int32)}
    inputs, _ = left_padded_batch(data, np.array([0, 1]), pad_token=0, device=torch.device("cpu"))
    with torch.inference_mode():
        last = last_logits(model, inputs)
        full = model(**inputs, use_cache=False).logits[:, -1, :].float()
    assert last.shape == (2, 64)
    torch.testing.assert_close(last, full, rtol=1e-5, atol=1e-5)


def test_cosine_schedule_matches_frozen_recipe():
    import math
    steps, warmup = CONFIG["steps"], CONFIG["warmup_steps"]
    def scale(current):
        progress = (current + 1 - warmup) / max(1, steps - warmup)
        return (current + 1) / warmup if current < warmup else 0.5 * (1.0 + math.cos(math.pi * progress))
    assert scale(0) == pytest.approx(1 / warmup) and scale(warmup - 1) == pytest.approx(1.0)
    assert scale(steps - 1) == pytest.approx(0.0, abs=1e-12)   # known: the final update runs at lr 0
    assert all(scale(i) >= scale(i + 1) for i in range(warmup, steps - 1))


def test_frozen_weights_skip_metadata_arrays_and_validate_vectors():
    selection = {"domain": np.array([0, 1, 1]), "bucket": np.array([0, 0, 1]), "pack_index": np.array([0, 1, 1]),
                 "stratum_uniform": np.full(3, 1 / 3), "direct_zipfit": np.array([0.0, 0.5, 0.5])}
    weights = frozen_weights(selection, 3)
    assert set(weights) == {"stratum_uniform", "direct_zipfit"}
    np.testing.assert_allclose(weights["direct_zipfit"], [0.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="Invalid frozen weights: bad"):
        frozen_weights({"bad": np.array([0.5, 0.6, 0.0])}, 3)
    with pytest.raises(ValueError, match="no selector vectors"):
        frozen_weights({"domain": np.array([0, 1, 1])}, 3)


@needs_data
def test_frozen_selection_file_loads_as_nine_valid_selectors():
    weights = frozen_weights(load("selection"), 9797)
    assert len(weights) == 9 and all(abs(w.sum() - 1.0) < 1e-9 for w in weights.values())
