"""Fixed inputs and pure utilities for Experiment 07 (training calibration).

Record identity, prompts and leakage scans are imported unchanged from
Experiment 05 version 2 so that every reconstructed identity matches the
frozen parent manifests byte for byte.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

HOME = Path(__file__).resolve().parent
EXPERIMENTS = HOME.parents[1]
EXP05 = EXPERIMENTS / "05_validation_guided_sft"


def load_module(path: Path, name: str, common: Path | None = None):
    """Import a sibling experiment's file without letting its `common` clash."""
    saved = sys.modules.get("common")
    try:
        if common is not None:
            spec = importlib.util.spec_from_file_location("common", common)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["common"] = module
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = saved


PARENT = load_module(EXP05 / "expt_v2" / "common.py", "exp05_v2_common")
canonical_record = PARENT.canonical_record
content_grams = PARENT.content_grams
stable_hash = PARENT.stable_hash
atomic_json = PARENT.atomic_json
file_hash = PARENT.file_hash
sha256 = PARENT.sha256
now = PARENT.now
LABELS = PARENT.LABELS
MODEL = PARENT.MODEL
MODEL_REVISION = PARENT.MODEL_REVISION
SOURCES = PARENT.SOURCES
V1_CONFIG = load_module(EXP05 / "expt_v1" / "common.py", "exp05_v1_common").CONFIG
V2_CONFIG = PARENT.CONFIG
V1_MANIFEST = EXP05 / "expt_v1" / "data_manifest.json"
V2_MANIFEST = EXP05 / "expt_v2" / "data_manifest.json"

# Frozen before any Experiment 07 data was scored or any model was trained.
SALT = "zip-mix/exp07/v1"
SPLIT = {
    "holdout_target": ("sciq", 500),        # Stage B fresh target; unread in Stage A
    "holdout_retention": ("commonsense_qa", 500),
    "calibration_target": ("sciq", 500),    # Stage A tuning validation
    "calibration_retention": ("commonsense_qa", 500),
    "calibration_dev_views": ("sciq", 400), # records packed into 8 x 4,096-byte views; diagnostics only
    "calibration_train_cap_per_domain": 3000,
}
# ARC-Challenge training records are exhausted by Experiment 05 (1,116/1,116).
# Its unused official validation split supplies the fresh ARC calibration rows.
ARC_FRESH_SPLIT = "validation"
STAGE_A = {
    "learning_rates": [2e-5, 1e-5, 5e-6, 2e-6],
    "reference_learning_rate": 2e-5,
    "seeds": [0, 1, 2, 3, 4],
    "retention_tolerance": .01,
    "tie_tolerance": .002,
    "minimum_seeds_at_or_above_base": 3,
    "minimum_prediction_change": .02,
}
STAGE_B = {
    "methods": ["conditional_signed_rank", "uniform_stratum", "permuted_stratum_rank"],
    "seeds": [0, 1, 2, 3, 4],
    "primary_seeds": [0, 1, 2],
    "tilt_log": float(np.log(4)), "permutation_seed": 1707,
    "advance_gain": .02, "minimum_positive_pairs": 4, "retention_tolerance": .01,
    "recipe_transfer_tolerance": .01,
    "admission": {"joint_mass_tolerance": 1e-12, "minimum_effective_fraction": .8,
                  "minimum_total_variation": .05},
}
TRAIN = {
    "steps": 128, "batch_size": 16, "weight_decay": .01, "warmup_steps": 8,
    "gradient_clip": 1.0, "max_input_length": 256, "checkpoint_every": 64,
    "max_infrastructure_resumes_per_cell": 1, "betas": [.9, .95], "eps": 1e-8,
}
EVALUATION = {"calibration": ["calibration_target", "calibration_retention"],
              "stage_b": ["holdout_target", "holdout_retention",
                          "calibration_target", "calibration_retention"]}


def salted_order(records, role):
    """Deterministic role-specific order, independent of earlier id ranking."""
    return sorted(records, key=lambda r: sha256(f"{SALT}:{role}:{r['id']}".encode()))


def overlaps(record, ids, questions, grams):
    return (record["id"] in ids or record["question_id"] in questions
            or bool(content_grams(record) & grams))


class Protected:
    """Exact-identity, normalized-question and 13-gram content guard."""

    def __init__(self, records=()):
        self.ids, self.questions, self.grams = set(), set(), set()
        self.add(records)

    def add(self, records):
        for r in records:
            self.ids.add(r["id"])
            self.questions.add(r["question_id"])
            self.grams.update(content_grams(r))

    def hits(self, record):
        return overlaps(record, self.ids, self.questions, self.grams)


def assign_roles(fresh, protected, order=None):
    """Greedy disjoint roles in frozen priority; later roles lose overlaps.

    fresh maps domain -> eligible records not used by earlier experiments.
    Returns role -> records and per-role exclusion counts. Every accepted
    record is added to the guard, so roles are pairwise exact- and
    13-gram-disjoint and disjoint from everything protected on entry.
    """
    order = order or ["holdout_target", "holdout_retention", "calibration_target",
                      "calibration_retention", "calibration_dev_views"]
    taken, roles, exclusions = set(), {}, {}
    for role in order:
        domain, cap = SPLIT[role]
        chosen, rejected = [], 0
        for record in salted_order(fresh.get(domain, []), role):
            if record["id"] in taken:
                continue
            if protected.hits(record):
                rejected += 1
                continue
            chosen.append(record)
            protected.add([record])
            taken.add(record["id"])
            if len(chosen) == cap:
                break
        if len(chosen) < cap:
            raise ValueError(f"Insufficient fresh records for {role}: {len(chosen)} < {cap}")
        roles[role], exclusions[role] = chosen, {"overlap_rejected": rejected}
    train, train_exclusions = [], {}
    for domain in sorted(fresh):
        chosen, rejected = [], 0
        for record in salted_order(fresh[domain], "calibration_train:" + domain):
            if record["id"] in taken:
                continue
            if protected.hits(record):
                rejected += 1
                continue
            chosen.append(record)
            taken.add(record["id"])
            if len(chosen) == SPLIT["calibration_train_cap_per_domain"]:
                break
        if not chosen:
            raise ValueError(f"No fresh calibration training rows in {domain}")
        # Training rows may share content with each other, never with any role.
        train.extend(chosen)
        train_exclusions[domain] = {"overlap_rejected": rejected, "chosen": len(chosen)}
    roles["calibration_train"] = sorted(train, key=lambda r: (r["domain"], r["id"]))
    exclusions["calibration_train"] = train_exclusions
    return roles, exclusions


def stage_b_probabilities(source, bucket, pack, contrast, seed=1707, tilt=float(np.log(4))):
    """Fixed joint source x bin mass; only within-stratum pack ranking changes.

    Reimplements NEXT_STAGE_DESIGN.md exactly (and next_stage_diagnostics.py):
    z=(R-0.5)/m-0.5 from average ranks of the signed contrast inside each
    stratum, w=exp(log(4) z) inherited by pack members, and one frozen
    within-stratum permutation for the shuffled control.
    """
    from scipy.stats import rankdata
    source, bucket, pack = map(np.asarray, (source, bucket, pack))
    contrast = np.asarray(contrast, dtype=float)
    npacks = len(contrast)
    pack_source = np.array([np.unique(source[pack == i]).item() for i in range(npacks)])
    pack_bucket = np.array([np.unique(bucket[pack == i]).item() for i in range(npacks)])
    rank, shuffled = np.zeros(npacks), np.zeros(npacks)
    rng = np.random.default_rng(seed)
    masks = []
    for s, b in sorted(set(zip(pack_source.tolist(), pack_bucket.tolist()))):
        indexes = np.flatnonzero((pack_source == s) & (pack_bucket == b))
        values = (rankdata(contrast[indexes], method="average") - .5) / len(indexes) - .5
        rank[indexes], shuffled[indexes] = values, rng.permutation(values)
        masks.append((source == s) & (bucket == b))

    def conditional(scores):
        weights = np.exp(tilt * scores[pack])
        p = np.zeros(len(pack))
        for rows in masks:
            p[rows] = rows.mean() * weights[rows] / weights[rows].sum()
        return p

    return {"conditional_signed_rank": conditional(rank),
            "uniform_stratum": np.full(len(pack), 1 / len(pack)),
            "permuted_stratum_rank": conditional(shuffled)}, masks


def admission_checks(probabilities, masks, source, bucket, config=None):
    """Pre-training engineering gates from NEXT_STAGE_DESIGN.md; no outcomes."""
    config = config or STAGE_B["admission"]
    source, bucket = np.asarray(source), np.asarray(bucket)
    out = {"joint_mass_max_error": 0.0, "effective_fraction": {}, "total_variation": {}}
    for name, p in probabilities.items():
        if not (np.isfinite(p).all() and (p > 0).all() and abs(p.sum() - 1) < config["joint_mass_tolerance"]):
            raise ValueError(f"Invalid probabilities: {name}")
        for rows in masks:
            out["joint_mass_max_error"] = max(out["joint_mass_max_error"], abs(p[rows].sum() - rows.mean()))
        for labels in (source, bucket):
            for label in np.unique(labels):
                rows = labels == label
                out["joint_mass_max_error"] = max(out["joint_mass_max_error"], abs(p[rows].sum() - rows.mean()))
        out["effective_fraction"][name] = float(1 / (p @ p) / len(p))
    tv = lambda a, b: float(np.abs(probabilities[a] - probabilities[b]).sum() / 2)
    aligned = "conditional_signed_rank"
    for other in ("uniform_stratum", "permuted_stratum_rank"):
        out["total_variation"][f"{aligned}_vs_{other}"] = tv(aligned, other)
    out["total_variation"]["uniform_stratum_vs_permuted_stratum_rank"] = tv("uniform_stratum", "permuted_stratum_rank")
    out["passed"] = bool(out["joint_mass_max_error"] <= config["joint_mass_tolerance"]
                         and min(out["effective_fraction"].values()) >= config["minimum_effective_fraction"]
                         and min(v for k, v in out["total_variation"].items() if k.startswith(aligned)) >= config["minimum_total_variation"])
    return out
