"""Exercise the cross-study audit against isolated copies of published proof."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from compare_completed_studies import contrast, load_study


@pytest.fixture
def study(tmp_path):
    source = Path(__file__).parent / "expt_v1"
    target = tmp_path / "expt_v1"
    target.mkdir()
    for name in ("data_manifest.json", "common.py", "train_sft.py", "analyze_sft.py"):
        shutil.copyfile(source / name, target / name)
    shutil.copytree(source / "results/measured", target / "results/measured")
    return tmp_path


def rewrite_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


def test_complete_reference_and_known_sign_tests(study):
    actual = load_study(study, "expt_v1", 9)
    assert len(actual[2]) == 9
    assert actual[4]["target"] == 404 / 500
    positive = contrast([.6, .65, .7], [.5, .55, .6])
    assert positive["mean"] == pytest.approx(.1)
    assert positive["p_value"] == .25
    assert contrast([.6, .4, .5], [.5, .5, .5])["p_value"] == 1


def test_incomplete_matrix_is_not_analyzed(study):
    path = study / "expt_v1/results/measured/summary.json"
    rewrite_json(path, lambda value: value.update(full_procedure_complete=False))
    with pytest.raises(ValueError, match="complete matrices"):
        load_study(study, "expt_v1", 9)


def test_changed_frozen_source_is_not_analyzed(study):
    path = study / "expt_v1/analyze_sft.py"
    path.write_text(path.read_text() + "\n# isolated corruption fixture\n")
    with pytest.raises(ValueError, match="source changed"):
        load_study(study, "expt_v1", 9)


def test_changed_reported_accuracy_is_rejected(study):
    path = study / "expt_v1/results/measured/contrastive_zipmix__seed0/result.json"
    rewrite_json(path, lambda value: value["metrics"]["target"].update(accuracy=.99))
    with pytest.raises(ValueError, match="Accuracy differs"):
        load_study(study, "expt_v1", 9)


def test_changed_gold_or_correctness_is_rejected(study):
    path = study / "expt_v1/results/measured/contrastive_zipmix__seed0/target_predictions.npz"
    with np.load(path, allow_pickle=False) as arrays:
        copied = {name: arrays[name].copy() for name in arrays.files}
    copied["gold"][0] = (copied["prediction"][0] + 1) % 4
    copied["correct"][0] = 1
    np.savez_compressed(path, **copied)
    with pytest.raises(ValueError, match="inconsistent accuracy"):
        load_study(study, "expt_v1", 9)
