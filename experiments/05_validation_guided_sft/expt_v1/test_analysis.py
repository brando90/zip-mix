"""Tests use invented fixtures only; no synthetic metric is a research result."""
import json

import numpy as np
import pytest

from analyze_sft import analyze, mean_interval, paired_estimate
from common import CONFIG, METHODS, SEEDS


def test_paired_seeds_not_evaluation_items():
    value = paired_estimate({0: .7, 1: .72, 2: .71}, {0: .6, 1: .61, 2: .62})
    assert value["n"] == 3
    assert value["paired_seeds"] == [0, 1, 2]
    assert value["p_value"] == .25
    assert value["mean"] == pytest.approx(.10)
    assert value["interval95"][0] < value["mean"] < value["interval95"][1]
    assert value["method_mean"] == pytest.approx(.71)
    assert value["baseline_mean"] == pytest.approx(.61)


def test_partial_pairs_have_no_interval_or_filled_values():
    value = paired_estimate({0: .7, 1: .6, 2: .8}, {0: .6, 2: .7})
    assert value["n"] == 2
    assert value["paired_seeds"] == [0, 2]
    assert value["interval95"] is None
    assert value["p_value"] == .5
    empty = paired_estimate({0: .8}, {1: .7})
    assert empty["n"] == 0
    assert empty["mean"] is None and empty["p_value"] is None


def test_sign_test_ties_and_direction():
    tie = paired_estimate({0: .7, 1: .7, 2: .7}, {0: .7, 1: .7, 2: .7})
    assert tie["ties"] == 3 and tie["p_value"] == 1
    mixed = paired_estimate({0: .7, 1: .8, 2: .9}, {0: .6, 1: .8, 2: 1})
    assert mixed["ties"] == 1 and mixed["p_value"] == 1
    negative = paired_estimate({0: .5, 1: .5, 2: .5}, {0: .6, 1: .6, 2: .6})
    assert negative["mean"] < 0 and negative["p_value"] == .25


def test_interval_rejects_nonfinite_and_omits_small_n():
    with pytest.raises(ValueError):
        mean_interval([1, np.nan, 3])
    with pytest.raises(ValueError):
        mean_interval([[1, 2]])
    assert mean_interval([.3])["interval95"] is None
    assert mean_interval([.3, .4])["interval95"] is None
    constant = mean_interval([.3, .3, .3])
    assert constant["interval95"] == pytest.approx([.3, .3])


def write_fixture(run, complete_keys=(), base_complete=False):
    run.mkdir(exist_ok=True)
    ledger = {"run_id": "synthetic-test-only", "identity": {"manifest_id": "synthetic-data-only"},
              "expected_training_cells": 18, "base": {"status": "complete" if base_complete else "pending"},
              "cells": [{"id": f"{m}__seed{s}", "method": m, "seed": s,
                         "status": "complete" if (m, s) in complete_keys else "pending"}
                        for m in METHODS for s in SEEDS]}
    for cell in ledger["cells"]:
        if cell["status"] == "complete":
            write_receipt(run / cell["id"], ledger, cell, 250 + 5 * METHODS.index(cell["method"]) + cell["seed"])
    if base_complete:
        write_receipt(run / "base", ledger, None, 200)
    (run / "ledger.json").write_text(json.dumps(ledger))
    return ledger


def write_receipt(path, ledger, cell, count):
    path.mkdir(parents=True, exist_ok=True)
    n = CONFIG["evaluation_cap_per_split"]
    metrics = {}
    for split in ("target", "retention"):
        gold = np.zeros(n, dtype=np.int32)
        predictions = np.ones(n, dtype=np.int32)
        predictions[:count] = 0
        correct = predictions == gold
        ids = np.array([f"{split}-{i}" for i in range(n)])
        np.savez(path / f"{split}_predictions.npz", ids=ids, correct=correct,
                 prediction=predictions, gold=gold, label_nll=np.ones(n))
        metrics[split] = {"count": n, "correct": count, "accuracy": count / n, "label_nll": 1.0}
    receipt = {"run_id": ledger["run_id"], "metrics": metrics}
    if cell is not None:
        receipt.update(cell=cell["id"], method=cell["method"], seed=cell["seed"],
                       manifest_id=ledger["identity"]["manifest_id"], steps=CONFIG["steps"],
                       supervised_tokens=CONFIG["steps"] * CONFIG["batch_size"],
                       processed_input_tokens=12345, training_seconds=12.3, infrastructure_resumes=[])
    (path / "result.json").write_text(json.dumps(receipt))


def test_complete_eighteen_cell_matrix_and_base_once(tmp_path):
    write_fixture(tmp_path, {(m, s) for m in METHODS for s in SEEDS}, base_complete=True)
    summary = analyze(tmp_path)
    assert summary["expected_training_cells"] == 18
    assert summary["verified_training_cells"] == 18
    assert summary["full_procedure_complete"]
    assert summary["base"]["target"]["count"] == 500
    assert summary["base"]["target"]["correct"] == 200
    assert len(summary["cells"]) == 18
    result = summary["zipmix_minus_baseline"]["sample_proportional"]["target"]
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(.03)
    assert result["p_value"] == .25
    assert summary["methods"]["zipmix_static"]["target"]["accuracy"]["n"] == 3
    text = (tmp_path / "results_summary.md").read_text()
    assert text.count("Base checkpoint: one deterministic evaluation") == 1
    assert "minimum p-value is 0.25" in text
    assert "Verified training cells: 18/18" in text
    assert json.loads((tmp_path / "summary.json").read_text())["full_procedure_complete"]


def test_pending_failed_and_missing_receipt_all_retained(tmp_path):
    ledger = write_fixture(tmp_path, {("sample_proportional", 0), ("zipmix_static", 0)})
    missing = tmp_path / "zipmix_static__seed0" / "result.json"
    missing.unlink()
    ledger["cells"][1]["status"] = "failed"
    # Entirely absent ledger row is synthesized as explicit missing, never dropped.
    ledger["cells"].pop()
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    summary = analyze(tmp_path)
    assert summary["expected_training_cells"] == len(summary["cells"]) == 18
    assert summary["verified_training_cells"] == 1
    assert summary["status_counts"]["failed"] == 1
    assert summary["status_counts"]["missing_ledger_row"] == 1
    assert summary["status_counts"]["invalid_or_missing_receipt"] == 1
    bad = next(c for c in summary["cells"] if c["id"] == "zipmix_static__seed0")
    assert bad["target_accuracy"] is None
    assert summary["zipmix_minus_baseline"]["sample_proportional"]["target"]["n"] == 0
    assert not summary["full_procedure_complete"]


@pytest.mark.parametrize("corruption", ["run", "accuracy", "token_budget", "gold", "missing_predictions"])
def test_invalid_receipt_is_explicit_not_silently_accepted(tmp_path, corruption):
    write_fixture(tmp_path, {("sample_proportional", 0)})
    path = tmp_path / "sample_proportional__seed0"
    receipt_path = path / "result.json"
    receipt = json.loads(receipt_path.read_text())
    if corruption == "run":
        receipt["run_id"] = "another-run"
    elif corruption == "accuracy":
        receipt["metrics"]["target"]["accuracy"] = .99
    elif corruption == "token_budget":
        receipt["supervised_tokens"] -= 1
    elif corruption == "gold":
        array_path = path / "target_predictions.npz"
        with np.load(array_path) as archive:
            arrays = {k: archive[k] for k in archive.files}
        arrays["gold"][0] = 4
        np.savez(array_path, **arrays)
    else:
        (path / "target_predictions.npz").unlink()
    receipt_path.write_text(json.dumps(receipt))
    summary = analyze(tmp_path)
    assert summary["verified_training_cells"] == 0
    assert len(summary["audit_errors"]) == 1
    assert summary["cells"][0]["analysis_status"] == "invalid_or_missing_receipt"


def test_duplicate_ledger_cell_rejected(tmp_path):
    ledger = write_fixture(tmp_path)
    ledger["cells"].append(ledger["cells"][0])
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="duplicated"):
        analyze(tmp_path)
