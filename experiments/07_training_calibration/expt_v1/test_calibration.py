"""Deterministic Experiment 07 tests; CPU only, no pretrained weights or network.

The frozen-pool test runs only when a private data/ snapshot exists.
"""
from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import calib_common as cc
from calib_common import (Protected, STAGE_A, STAGE_B, TRAIN, admission_checks, assign_roles,
                          canonical_record, file_hash, salted_order, stable_hash, stage_b_probabilities)
import analyze_calibration as an
import train_calibration as tr

LABEL_TOKENS = [362, 425, 356, 422, 468]
PAD = 151643


def item(text, domain="sciq", split="train"):
    words = text.split()
    row = dict(question=text, correct_answer=f"right {words[0]}", distractor1=f"wrong one {words[-1]}",
               distractor2="wrong two", distractor3="wrong three")
    if domain == "sciq":
        return canonical_record(row, "sciq", split)
    return canonical_record(dict(question=text, choices={"label": ["A", "B", "C"], "text": [f"x {words[0]}", "y", "z"]},
                                 answerKey="A"), domain, split)


def unique_questions(n, domain, tag):
    return [item(f"{tag} question number {i} about topic {i * 7} with distinct wording {domain}", domain) for i in range(n)]


# ---------------------------------------------------------------- split rules

def test_salted_order_is_deterministic_and_role_specific():
    rows = unique_questions(50, "sciq", "order")
    assert [r["id"] for r in salted_order(rows, "a")] == [r["id"] for r in salted_order(list(reversed(rows)), "a")]
    assert [r["id"] for r in salted_order(rows, "a")] != [r["id"] for r in salted_order(rows, "b")]


def small_split(monkeypatch):
    for role, value in {"holdout_target": ("sciq", 5), "holdout_retention": ("commonsense_qa", 5),
                        "calibration_target": ("sciq", 5), "calibration_retention": ("commonsense_qa", 5),
                        "calibration_dev_views": ("sciq", 3)}.items():
        monkeypatch.setitem(cc.SPLIT, role, value)
    monkeypatch.setitem(cc.SPLIT, "calibration_train_cap_per_domain", 6)


def test_roles_are_disjoint_capped_and_respect_protected(monkeypatch):
    small_split(monkeypatch)
    fresh = {"sciq": unique_questions(40, "sciq", "fresh"), "commonsense_qa": unique_questions(30, "commonsense_qa", "fresh")}
    prior = fresh["sciq"][:3]
    roles, exclusions = assign_roles(fresh, Protected(prior))
    ids = [r["id"] for rows in roles.values() for r in rows]
    assert len(ids) == len(set(ids))
    assert not {r["id"] for r in prior} & set(ids)
    assert len(roles["holdout_target"]) == 5 and len(roles["calibration_dev_views"]) == 3
    assert sum(r["domain"] == "sciq" for r in roles["calibration_train"]) == 6
    names = list(roles)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            guard = Protected(roles[b])
            assert not any(guard.hits(r) for r in roles[a]), (a, b)


def test_thirteen_gram_overlap_goes_to_the_earlier_role(monkeypatch):
    small_split(monkeypatch)
    shared = "the mitochondria is the powerhouse of the cell and it makes energy for every living tissue"
    a, b = item(shared + " alpha"), item("beta " + shared)
    assert a["id"] != b["id"] and Protected([a]).hits(b)
    fresh = {"sciq": [a, b] + unique_questions(40, "sciq", "pad"), "commonsense_qa": unique_questions(30, "commonsense_qa", "pad")}
    roles, _ = assign_roles(fresh, Protected())
    placed = [name for name, rows in roles.items() for r in rows if r["id"] in (a["id"], b["id"])]
    assert len(placed) == 1


def test_insufficient_fresh_records_raise(monkeypatch):
    small_split(monkeypatch)
    with pytest.raises(ValueError, match="Insufficient"):
        assign_roles({"sciq": unique_questions(7, "sciq", "few"), "commonsense_qa": unique_questions(30, "commonsense_qa", "x")}, Protected())


# ---------------------------------------------------------- Stage B mixtures

def synthetic_pool(seed=3):
    rng = np.random.default_rng(seed)
    pack_source = np.repeat(np.arange(3), 8)
    pack_bucket = np.tile(np.arange(4), 6)
    sizes = rng.integers(2, 6, len(pack_source))
    pack = np.repeat(np.arange(len(sizes)), sizes)
    return pack_source[pack], pack_bucket[pack], pack, rng.normal(size=len(sizes))


def test_stage_b_fixes_joint_mass_and_bounds_tilt():
    source, bucket, pack, contrast = synthetic_pool()
    p, masks = stage_b_probabilities(source, bucket, pack, contrast)
    for rows in masks:
        for name in p:
            assert abs(p[name][rows].sum() - rows.mean()) < 1e-12
        ratio = p["conditional_signed_rank"][rows].max() / p["conditional_signed_rank"][rows].min()
        assert ratio < 4
    assert np.allclose(p["uniform_stratum"], 1 / len(pack))
    again, _ = stage_b_probabilities(source, bucket, pack, contrast)
    assert all(np.array_equal(p[k], again[k]) for k in p)


def test_constant_scores_reduce_to_uniform():
    source, bucket, pack, _ = synthetic_pool()
    p, _ = stage_b_probabilities(source, bucket, pack, np.zeros(pack.max() + 1))
    assert np.allclose(p["conditional_signed_rank"], 1 / len(pack))


def test_admission_rejects_an_inert_selector():
    source, bucket, pack, _ = synthetic_pool()
    p, masks = stage_b_probabilities(source, bucket, pack, np.zeros(pack.max() + 1))
    assert not admission_checks(p, masks, source, bucket)["passed"]


DATA = Path(__file__).parent / "data" / "stage_b"


@pytest.mark.skipif(not (DATA / "selection.npz").exists(), reason="private frozen snapshot absent")
def test_frozen_pool_reproduces_next_stage_design():
    receipt = json.loads((Path(__file__).parents[2] / "04_validation_guided_zipmix/expt_v1/results/next_stage_diagnostics.json").read_text())
    with np.load(DATA / "selection.npz") as s:
        assert abs(1 / (s["conditional_signed_rank"] @ s["conditional_signed_rank"]) - receipt["metrics"]["aligned"]["effective_sample_size"]) < 1e-6


# ------------------------------------------------------ predeclared decisions

def stage_a_cells(target, retention, change=.1, base_pred=None):
    cells = []
    for lr in STAGE_A["learning_rates"]:
        for s in STAGE_A["seeds"]:
            pred = np.array(base_pred)
            flip = int(change * len(pred))
            pred[:flip] = 1 - pred[:flip]
            cells.append({"learning_rate": lr, "seed": s, "status": "complete",
                          "calibration_target": {"accuracy": target[lr] + .001 * s - .002, "prediction": pred.tolist()},
                          "calibration_retention": {"accuracy": retention[lr]}})
    return cells


BASE_PRED = [0] * 500
BASE = {"calibration_target": {"accuracy": .80, "prediction": BASE_PRED}, "calibration_retention": {"accuracy": .57}}


def test_stage_a_selects_best_retention_eligible_rate():
    target = {2e-5: .84, 1e-5: .82, 5e-6: .81, 2e-6: .80}
    retention = {2e-5: .54, 1e-5: .565, 5e-6: .57, 2e-6: .57}
    result = an.stage_a_decision(BASE, stage_a_cells(target, retention, base_pred=BASE_PRED))
    assert result["selected_learning_rate"] == 1e-5 and result["gate"]["passed"]


def test_stage_a_tie_goes_to_larger_rate_and_gate_fails_below_base():
    target = {2e-5: .79, 1e-5: .79, 5e-6: .789, 2e-6: .70}
    retention = {k: .57 for k in target}
    result = an.stage_a_decision(BASE, stage_a_cells(target, retention, base_pred=BASE_PRED))
    assert result["selected_learning_rate"] == 2e-5
    assert not result["gate"]["passed"] and not result["gate"]["checks"]["target_mean_at_or_above_base"]


def test_stage_a_rejects_an_inert_recipe():
    target = {k: .80 for k in STAGE_A["learning_rates"]}
    retention = {k: .57 for k in target}
    result = an.stage_a_decision(BASE, stage_a_cells({k: .802 for k in target}, retention, change=0, base_pred=BASE_PRED))
    assert not result["gate"]["checks"]["recipe_moves_predictions"] and not result["gate"]["passed"]


def test_stage_a_incomplete_grid_gives_no_decision():
    target = {k: .85 for k in STAGE_A["learning_rates"]}
    cells = stage_a_cells(target, {k: .57 for k in target}, base_pred=BASE_PRED)
    cells[3]["status"] = "failed"
    result = an.stage_a_decision(BASE, cells)
    assert not result["complete"] and not result["gate"]["passed"] and "selected_learning_rate" not in result


def stage_b_cells(gains):
    cells = []
    for m in STAGE_B["methods"]:
        for s in STAGE_B["seeds"]:
            cells.append({"method": m, "seed": s, "status": "complete",
                          "holdout_target": {"accuracy": .80 + gains[m][s]},
                          "holdout_retention": {"accuracy": .57}, "calibration_target": {"accuracy": .81}})
    return cells


STAGE_B_BASE = {"holdout_target": {"accuracy": .80}, "holdout_retention": {"accuracy": .57},
                "calibration_target": {"accuracy": .80}, "calibration_retention": {"accuracy": .57}}


def test_stage_b_advance_rule():
    gains = {"conditional_signed_rank": [.03, .04, .03, .05, .02], "uniform_stratum": [0] * 5, "permuted_stratum_rank": [.01] * 5}
    result = an.stage_b_analysis(STAGE_B_BASE, stage_b_cells(gains))
    assert result["advance"] and result["recipe_transfer_ok"]
    assert result["contrasts"]["uniform_stratum"]["target"]["sign_p"] == pytest.approx(.0625)
    gains["conditional_signed_rank"][0] = -.05
    assert not an.stage_b_analysis(STAGE_B_BASE, stage_b_cells(gains))["advance"]  # mean gain 1.8 < 2 points
    gains["conditional_signed_rank"] = [.03, -.01, -.01, .05, .02]
    assert not an.stage_b_analysis(STAGE_B_BASE, stage_b_cells(gains))["advance"]


def test_stage_b_incomplete_is_not_a_result():
    gains = {m: [0] * 5 for m in STAGE_B["methods"]}
    cells = stage_b_cells(gains)
    cells[0]["status"] = "failed"
    result = an.stage_b_analysis(STAGE_B_BASE, cells)
    assert not result["complete"] and not result["advance"]


def test_sign_test_and_holm():
    assert an.sign_test([1, 1, 1]) == pytest.approx(.25)
    assert an.sign_test([1] * 5) == pytest.approx(.0625)
    assert an.sign_test([0, 0]) == 1.0
    assert an.holm([.01, .04]) == pytest.approx([.02, .04])


# ------------------------------------------------------------- trainer pieces

def test_schedule_matches_experiment05():
    import math
    steps, warmup = TRAIN["steps"], TRAIN["warmup_steps"]
    for i in (0, 7, 8, 64, 127):
        progress = (i + 1 - warmup) / max(1, steps - warmup)
        expected = (i + 1) / warmup if i < warmup else .5 * (1 + math.cos(math.pi * progress))
        assert tr.learning_rate_scale(i, steps, warmup) == pytest.approx(expected)
    assert tr.learning_rate_scale(steps - 1, steps, warmup) == pytest.approx(0, abs=1e-12)


def tiny_model():
    return tr.new_model(Namespace(smoke=True, cache_dir=None), torch.device("cpu"), 0).eval()


def test_left_padding_matches_unpadded_last_logits():
    model = tiny_model()
    data = {"input_ids": np.array([[5, 6, 7, 8], [9, 10, 0, 0]]), "lengths": np.array([4, 2])}
    batch, count = tr.left_padded_batch(data, np.array([0, 1]), PAD, torch.device("cpu"))
    assert count == 6 and batch["position_ids"][1, -2:].tolist() == [0, 1]
    with torch.no_grad():
        padded = tr.last_logits(model, batch)
        alone = model(input_ids=torch.tensor([[9, 10]])).logits[:, -1, :]
    assert torch.allclose(padded[1], alone[0], atol=1e-4)


def fake_split(directory, stage, n_train=64, n_eval=24):
    rng = np.random.default_rng(0)
    directory.mkdir(parents=True)

    def arrays(n, tag):
        lengths = rng.integers(5, 20, n)
        ids = np.full((n, TRAIN["max_input_length"] - 1), PAD, dtype=np.int32)
        for i, length in enumerate(lengths):
            ids[i, :length] = rng.integers(10, 1000, length)
        answer = rng.integers(0, 4, n)
        return dict(input_ids=ids, lengths=lengths.astype(np.int32), answer=answer.astype(np.int32),
                    answer_token=np.asarray(LABEL_TOKENS)[answer].astype(np.int32), nchoices=np.full(n, 4, dtype=np.int32),
                    ids=np.array([f"{tag}{i}" for i in range(n)]), question_ids=np.array([f"q{tag}{i}" for i in range(n)]),
                    domain=np.array(["sciq"] * n))

    np.savez_compressed(directory / "train.npz", **arrays(n_train, "t"))
    sets = cc.EVALUATION[stage]
    for name in sets:
        np.savez_compressed(directory / f"{name}.npz", **arrays(n_eval, name[:3]))
    if stage == "stage_b":
        source = np.repeat([0, 1], n_train // 2)
        pack = np.arange(n_train) // 4
        bucket = pack % 2
        p, _ = stage_b_probabilities(source, bucket, pack, rng.normal(size=pack.max() + 1))
        np.savez_compressed(directory / "selection.npz", domain=source, bucket=bucket, pack_index=pack, **p)
    manifest = {"stage": stage, "evaluation_sets": sets, "model": cc.MODEL, "model_revision": cc.MODEL_REVISION,
                "train_config": TRAIN, "pad_token_id": PAD, "label_tokens": LABEL_TOKENS,
                "files": {p.name: file_hash(p) for p in sorted(directory.iterdir())}}
    manifest["manifest_id"] = stable_hash(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def run_args(data, output, decision=None):
    return Namespace(data=data, output=output, decision=decision, cache_dir=None, eval_batch_size=8,
                     save_models=False, smoke=True)


def test_smoke_run_both_stages_and_decision_chain(tmp_path):
    calib = fake_split(tmp_path / "calibration", "calibration")
    (tmp_path / "a").mkdir()
    assert tr.run(run_args(calib, tmp_path / "a")) == "complete"
    ledger, base, cells = an.load_run(tmp_path / "a")
    config = {**STAGE_A, "learning_rates": sorted({c["learning_rate"] for c in cells}), "seeds": [0]}
    decision = an.stage_a_decision(base, cells, config)
    decision.update(run_id=ledger["run_id"], smoke=True)
    decision["decision_id"] = stable_hash(decision)
    (tmp_path / "a" / "decision.json").write_text(json.dumps(decision))
    stage_b = fake_split(tmp_path / "stage_b", "stage_b")
    (tmp_path / "b").mkdir()
    assert tr.run(run_args(stage_b, tmp_path / "b", tmp_path / "a" / "decision.json")) == "complete"
    receipt = json.loads((tmp_path / "b" / "conditional_signed_rank__seed0" / "result.json").read_text())
    assert set(receipt["metrics"]) == set(cc.EVALUATION["stage_b"]) and receipt["steps"] == tr.SMOKE["steps"]


def test_stage_b_refuses_missing_tampered_or_failed_decision(tmp_path):
    stage_b = fake_split(tmp_path / "stage_b", "stage_b")
    (tmp_path / "b").mkdir()
    with pytest.raises(SystemExit, match="requires --decision"):
        tr.run(run_args(stage_b, tmp_path / "b"))
    decision = {"gate": {"passed": False}, "selected_learning_rate": 1e-5, "smoke": False}
    decision["decision_id"] = stable_hash(decision)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision))
    with pytest.raises(SystemExit, match="not admitted"):
        tr.verified_decision(path, smoke=False)
    decision["selected_learning_rate"] = 2e-5
    path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="identity"):
        tr.verified_decision(path, smoke=False)


def test_changed_frozen_input_is_refused(tmp_path):
    calib = fake_split(tmp_path / "calibration", "calibration")
    with np.load(calib / "train.npz") as a:
        arrays = {k: a[k] for k in a.files}
    arrays["answer"][0] = (arrays["answer"][0] + 1) % 4
    np.savez_compressed(calib / "train.npz", **arrays)
    with pytest.raises(ValueError, match="Frozen input changed"):
        tr.verified_manifest(calib)


def test_interrupted_cell_resumes_once_to_identical_result(tmp_path, monkeypatch):
    calib = fake_split(tmp_path / "calibration", "calibration")
    (tmp_path / "clean").mkdir()
    tr.run(run_args(calib, tmp_path / "clean"))
    clean = json.loads((tmp_path / "clean" / "lr2e-05__seed0" / "result.json").read_text())
    original = tr.atomic_torch

    def crash_after_checkpoint(path, value):
        original(path, value)
        raise SystemExit("simulated preemption")

    monkeypatch.setattr(tr, "atomic_torch", crash_after_checkpoint)
    (tmp_path / "resumed").mkdir()
    with pytest.raises(SystemExit, match="preemption"):
        tr.run(run_args(calib, tmp_path / "resumed"))
    monkeypatch.setattr(tr, "atomic_torch", original)
    assert tr.run(run_args(calib, tmp_path / "resumed")) == "complete"
    resumed = json.loads((tmp_path / "resumed" / "lr2e-05__seed0" / "result.json").read_text())
    assert len(resumed["infrastructure_resumes"]) == 1
    assert resumed["metrics"] == clean["metrics"]
    assert resumed["processed_input_tokens"] == clean["processed_input_tokens"]
