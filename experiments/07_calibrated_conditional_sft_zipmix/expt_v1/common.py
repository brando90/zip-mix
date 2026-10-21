"""Fixed public inputs, within-stratum conditional alignment selectors, and Stage 1 calibration gate for Experiment 07."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from scipy.stats import rankdata

MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PARENT_V1_MANIFEST_ID = "d133f1242f8edb9827ddcd1c3ad404bac8b56f688943c0d943ff3063ce5e097a"
PARENT_V2_MANIFEST_ID = "0d6dec93e7d7ccffd87f95e7ebd19e78c84ec72a546007b122888789aba5cb44"
PARENT_EXP06_MANIFEST_ID = "242d063db24573adc525b7fb523cc72bc0b6a01cb7b4f36248b823f93bccc1a8"

SOURCES = {
    "sciq": ("allenai/sciq", None, "2c94ad3e1aafab77146f384e23536f97a4849815"),
    "openbookqa": ("allenai/openbookqa", "main", "388097ea7776314e93a529163e0fea805b8a6454"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453"),
    "commonsense_qa": ("tau/commonsense_qa", None, "94630fe30dad47192a8546eb75f094926d47e155"),
}

CALIBRATION_LRS = {
    "lr_1e6": 1e-6,
    "lr_2e6": 2e-6,
    "lr_5e6": 5e-6,
    "lr_1e5": 1e-5,
    "lr_2e5": 2e-5,
}
CALIBRATION_METHODS = tuple(CALIBRATION_LRS.keys())
CALIBRATION_SEEDS = (0, 1, 2)
CALIBRATION_RETENTION_DROP_TOLERANCE = 0.010

CONDITIONAL_METHODS = ("conditional_aligned", "stratum_uniform", "conditional_shuffled")
CONDITIONAL_SEEDS = (0, 1, 2, 3, 4, 5)

SELECTOR_METHODS = (
    "zipmix_static",
    "shuffled_zipmix",
    "wrongtarget_zipmix",
    "contrastive_zipmix",
    "source_matched",
    "direct_zipfit",
)
SELECTOR_SEEDS = (0, 1, 2)

ALL_STAGE2_METHODS = CONDITIONAL_METHODS + SELECTOR_METHODS
LABELS = "ABCDE"
TOLERANCE = 1e-12

CONFIG = {
    "steps": 128,
    "batch_size": 16,
    "weight_decay": 0.01,
    "warmup_steps": 8,
    "gradient_clip": 1.0,
    "max_input_length": 256,
    "training_cap_per_domain": 3000,
    "evaluation_cap_per_split": 500,
    "test_holdout2_count": 490,
    "alignment_views": 8,
    "score_byte_window": 4096,
    "gzip_level": 6,
    "compression_bins": 5,
    "wrongtarget_reserved_records": 256,
    "minimum_matched_target_views": 2,
    "direct_selection_fraction": 0.25,
    "checkpoint_every": 64,
    "max_infrastructure_resumes_per_cell": 1,
    "conditional_permutation_seed": 1707,
    "contrastive_permutation_seed": 1606,
    "calibration_retention_drop_tolerance": CALIBRATION_RETENTION_DROP_TOLERANCE,
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def grams13(text: str) -> set[str]:
    words = re.findall(r"\w+", text.casefold())
    return {sha256(" ".join(words[i:i + 13]).encode())
            for i in range(len(words) - 12)}


def content_grams(record: dict) -> set[str]:
    out = grams13(record["question"])
    for choice in record["choices"]:
        out.update(grams13(choice))
    return out


def canonical_record(row: dict, domain: str, split: str) -> dict:
    if domain == "sciq":
        question = row["question"]
        options = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
        answer = 0
    else:
        question = row["question_stem"] if domain == "openbookqa" else row["question"]
        options = list(row["choices"]["text"])
        answer = list(row["choices"]["label"]).index(row["answerKey"])
    question = " ".join(str(question).split())
    options = [" ".join(str(x).split()) for x in options]
    if not 2 <= len(options) <= len(LABELS) or not question or any(not x for x in options):
        raise ValueError("Invalid multiple-choice item")
    if len(set(normalize(x) for x in options)) != len(options):
        raise ValueError("Duplicate answer options")
    identity = stable_hash([normalize(question), sorted(normalize(x) for x in options)])
    order = sorted(range(len(options)), key=lambda i: sha256((identity + ":" + str(i)).encode()))
    options = [options[i] for i in order]
    answer = order.index(answer)
    content = question + "\n" + "\n".join(f"{LABELS[i]}. {x}" for i, x in enumerate(options))
    prompt = "Choose the correct answer.\nQuestion: " + content + "\nAnswer:"
    return {
        "id": identity,
        "question_id": sha256(normalize(question).encode()),
        "domain": domain,
        "split": split,
        "question": question,
        "choices": options,
        "answer": answer,
        "content": content,
        "prompt": prompt,
    }


def within_stratum_conditional_weights(
    source: np.ndarray,
    bucket: np.ndarray,
    pack_index: np.ndarray,
    true_max: np.ndarray,
    wrong_max: np.ndarray,
    permutation_seed: int = 1707,
) -> tuple[dict[str, np.ndarray], dict]:
    """Exact NEXT_STAGE_DESIGN.md conditional within-(source, bin) rank-tilted selector recipe."""
    source = np.asarray(source, dtype=np.int64)
    bucket = np.asarray(bucket, dtype=np.int64)
    pack_index = np.asarray(pack_index, dtype=np.int64)
    true_max = np.asarray(true_max, dtype=np.float64)
    wrong_max = np.asarray(wrong_max, dtype=np.float64)
    if source.ndim != 1 or not len(source) or source.shape != bucket.shape or source.shape != pack_index.shape:
        raise ValueError("Invalid example arrays")
    if true_max.ndim != 1 or true_max.shape != wrong_max.shape or not len(true_max):
        raise ValueError("Invalid pack score arrays")
    if not np.isfinite(true_max).all() or not np.isfinite(wrong_max).all():
        raise ValueError("Nonfinite pack scores")
    if set(pack_index.tolist()) != set(range(len(true_max))):
        raise ValueError("Every pack must be represented by valid pack_index")

    contrast = true_max - wrong_max
    pack_source = np.array([np.unique(source[pack_index == i]).item() for i in range(len(contrast))], dtype=np.int64)
    pack_bucket = np.array([np.unique(bucket[pack_index == i]).item() for i in range(len(contrast))], dtype=np.int64)
    rank = np.zeros(len(contrast), dtype=np.float64)
    shuffled = np.zeros(len(contrast), dtype=np.float64)
    rng = np.random.default_rng(permutation_seed)
    masks = []
    strata_info = []
    for s, b in sorted(set(zip(pack_source.tolist(), pack_bucket.tolist()))):
        indexes = np.flatnonzero((pack_source == s) & (pack_bucket == b))
        if len(indexes) <= 1 or len(np.unique(contrast[indexes])) <= 1:
            values = np.zeros(len(indexes), dtype=np.float64)
        else:
            values = (rankdata(contrast[indexes], method="average") - 0.5) / len(indexes) - 0.5
        perm_values = rng.permutation(values)
        if not np.array_equal(np.sort(values), np.sort(perm_values)):
            raise ValueError("Pack-rank permutation changed multiset")
        rank[indexes] = values
        shuffled[indexes] = perm_values
        for arr in (values, perm_values):
            w = np.exp(np.log(4.0) * arr)
            if w.max() / w.min() > 4.0 + TOLERANCE:
                raise ValueError("Within-stratum weight ratio exceeded 4")
        rows = (source == s) & (bucket == b)
        masks.append(rows)
        strata_info.append({
            "source_id": int(s),
            "bin": int(b),
            "packs": int(len(indexes)),
            "questions": int(rows.sum()),
            "fixed_joint_mass": float(rows.mean()),
        })

    def _normalize_within_strata(pack_z: np.ndarray) -> np.ndarray:
        weights = np.exp(np.log(4.0) * pack_z[pack_index])
        p = np.zeros(len(pack_index), dtype=np.float64)
        for rows in masks:
            p[rows] = rows.mean() * weights[rows] / weights[rows].sum()
        return p

    out = {
        "conditional_aligned": _normalize_within_strata(rank),
        "stratum_uniform": np.full(len(pack_index), 1.0 / len(pack_index), dtype=np.float64),
        "conditional_shuffled": _normalize_within_strata(shuffled),
    }
    max_joint_err = 0.0
    for name, p in out.items():
        if not np.isfinite(p).all() or not (p > 0).all() or abs(p.sum() - 1.0) >= TOLERANCE:
            raise ValueError(f"Invalid probability vector: {name}")
        for rows in masks:
            err = abs(float(p[rows].sum()) - float(rows.mean()))
            if err >= TOLERANCE:
                raise ValueError(f"Joint mass discrepancy in {name}: {err}")
            max_joint_err = max(max_joint_err, err)
        if (1.0 / (p @ p)) / len(p) < 0.80:
            raise ValueError(f"Effective sample size below 80% gate for {name}")
    tv_u = float(0.5 * np.abs(out["conditional_aligned"] - out["stratum_uniform"]).sum())
    tv_s = float(0.5 * np.abs(out["conditional_aligned"] - out["conditional_shuffled"]).sum())
    if tv_u < 0.05 or tv_s < 0.05:
        raise ValueError("Total variation gate (< 0.05) failed")
    diagnostics = {
        "nonempty_strata": len(strata_info),
        "maximum_joint_mass_absolute_error": max_joint_err,
        "tv_aligned_vs_uniform": tv_u,
        "tv_aligned_vs_shuffled": tv_s,
        "tv_uniform_vs_shuffled": float(0.5 * np.abs(out["stratum_uniform"] - out["conditional_shuffled"]).sum()),
        "effective_sample_fraction": {k: float((1.0 / (v @ v)) / len(v)) for k, v in out.items()},
    }
    return out, diagnostics


def select_calibrated_lr(
    method_metrics: dict[str, dict],
    base_metrics: dict,
    lrs: dict[str, float] = CALIBRATION_LRS,
    retention_drop_tol: float = CALIBRATION_RETENTION_DROP_TOLERANCE,
) -> dict:
    """Deterministic Stage 1 calibration gate using ONLY cal_target and cal_retention."""
    base_target_acc = float(base_metrics["cal_target"]["accuracy"])
    base_ret_acc = float(base_metrics["cal_retention"]["accuracy"])
    retention_floor = base_ret_acc - retention_drop_tol

    candidates = []
    for name, lr_val in lrs.items():
        if name not in method_metrics:
            raise ValueError(f"Missing calibration method: {name}")
        entry = method_metrics[name]
        t_acc = entry["cal_target"]["accuracy"]["mean"]
        r_acc = entry["cal_retention"]["accuracy"]["mean"]
        t_nll = entry["cal_target"].get("label_nll_mean")
        if t_acc is None or r_acc is None:
            raise ValueError(f"Incomplete calibration cells for {name}")
        feasible = bool(r_acc >= retention_floor - 1e-12)
        non_degrading_target = bool(t_acc >= base_target_acc - 1e-12)
        candidates.append({
            "method": name,
            "learning_rate": float(lr_val),
            "cal_target_accuracy_mean": float(t_acc),
            "cal_target_minus_base_pp": float((t_acc - base_target_acc) * 100.0),
            "cal_retention_accuracy_mean": float(r_acc),
            "cal_retention_minus_base_pp": float((r_acc - base_ret_acc) * 100.0),
            "cal_target_label_nll_mean": float(t_nll) if t_nll is not None else 0.0,
            "meets_retention_floor": feasible,
            "non_degrading_vs_base_target": non_degrading_target,
        })

    feasible_pool = [c for c in candidates if c["meets_retention_floor"]]
    if feasible_pool:
        chosen = max(
            feasible_pool,
            key=lambda c: (
                round(c["cal_target_accuracy_mean"], 12),
                round(-c["cal_target_label_nll_mean"], 12),
                -c["learning_rate"],
            ),
        )
        selection_reason = "maximized_cal_target_subject_to_retention_floor"
    else:
        chosen = max(
            candidates,
            key=lambda c: (
                round(c["cal_target_accuracy_mean"] + c["cal_retention_accuracy_mean"], 12),
                -c["learning_rate"],
            ),
        )
        selection_reason = "fallback_maximized_sum_no_candidate_met_retention_floor"

    return {
        "selected_method": chosen["method"],
        "selected_learning_rate": chosen["learning_rate"],
        "selection_reason": selection_reason,
        "base_cal_target_accuracy": base_target_acc,
        "base_cal_retention_accuracy": base_ret_acc,
        "retention_floor": retention_floor,
        "selected_cal_target_accuracy": chosen["cal_target_accuracy_mean"],
        "selected_cal_retention_accuracy": chosen["cal_retention_accuracy_mean"],
        "selected_cal_target_minus_base_pp": chosen["cal_target_minus_base_pp"],
        "selected_cal_retention_minus_base_pp": chosen["cal_retention_minus_base_pp"],
        "calibration_preserves_or_improves_base_target": chosen["non_degrading_vs_base_target"],
        "candidates": candidates,
    }
