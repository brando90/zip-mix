"""Fixed public inputs and pure utilities for the small SFT mechanism screen."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

import numpy as np

MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SOURCES = {
    "sciq": ("allenai/sciq", None, "2c94ad3e1aafab77146f384e23536f97a4849815"),
    "openbookqa": ("allenai/openbookqa", "main", "388097ea7776314e93a529163e0fea805b8a6454"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453"),
    "commonsense_qa": ("tau/commonsense_qa", None, "94630fe30dad47192a8546eb75f094926d47e155"),
}
METHODS = ("sample_proportional", "uniform_domain", "compel_filter", "zipmix_static",
           "direct_zipfit", "shuffled_zipmix", "wrongtarget_zipmix")
SEEDS = (0, 1, 2)
LABELS = "ABCDE"
CONFIG = {
    "steps": 128, "batch_size": 16, "learning_rate": 2e-5,
    "weight_decay": .01, "warmup_steps": 8, "gradient_clip": 1.0,
    "max_input_length": 256, "training_cap_per_domain": 3000,
    "evaluation_cap_per_split": 500, "alignment_views": 8,
    "score_byte_window": 4096, "gzip_level": 6, "compression_bins": 5,
    "wrongtarget_reserved_records": 256, "minimum_matched_target_views": 2,
    "direct_selection_fraction": .25, "checkpoint_every": 16,
    "max_infrastructure_resumes_per_cell": 1,
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
    """Question/choice content only; wrappers and answer labels are never scanned."""
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
    # A deterministic order independent of the correct label removes SciQ's
    # otherwise constant answer position without adding a model-derived choice.
    order = sorted(range(len(options)), key=lambda i: sha256((identity + ":" + str(i)).encode()))
    options = [options[i] for i in order]
    answer = order.index(answer)
    content = question + "\n" + "\n".join(f"{LABELS[i]}. {x}" for i, x in enumerate(options))
    prompt = "Choose the correct answer.\nQuestion: " + content + "\nAnswer:"
    return {"id": identity, "question_id": sha256(normalize(question).encode()),
            "domain": domain, "split": split, "question": question, "choices": options,
            "answer": answer, "content": content, "prompt": prompt}


def mixture_weights(domain, bucket, cr, scores, ids, direct_scores=None) -> tuple[dict[str, np.ndarray], dict]:
    domain, bucket = np.asarray(domain), np.asarray(bucket)
    cr, scores = np.asarray(cr, dtype=float), np.asarray(scores, dtype=float)
    n = len(domain)
    if not n or any(len(x) != n for x in (bucket, cr, scores, ids)):
        raise ValueError("Invalid selection input lengths")
    if not np.isfinite(scores).all() or not np.isfinite(cr).all():
        raise ValueError("Nonfinite compression statistics")
    if np.any(bucket < 0):
        raise ValueError("Invalid bucket index")
    score = np.maximum(scores, 0.0)
    result = {"sample_proportional": np.full(n, 1 / n)}
    domains, counts = np.unique(domain, return_counts=True)
    count_map = dict(zip(domains.tolist(), counts.tolist()))
    result["uniform_domain"] = np.array([1 / (len(domains) * count_map[d]) for d in domain])
    band = (cr >= .65) & (cr <= .80)
    if not band.any():
        raise ValueError("Compel band is empty; revise prospectively before launch")
    result["compel_filter"] = band.astype(float) / band.sum()

    def by_bucket(values):
        masses = np.bincount(bucket, weights=values)
        sizes = np.bincount(bucket)
        if masses.sum() <= 0:
            return np.full(n, 1 / n)
        return masses[bucket] / masses.sum() / sizes[bucket]

    result["zipmix_static"] = by_bucket(score)
    top_k = max(1, int(np.ceil(CONFIG["direct_selection_fraction"] * n)))
    direct_scores = scores if direct_scores is None else np.asarray(direct_scores, dtype=float)
    if len(direct_scores) != n or not np.isfinite(direct_scores).all():
        raise ValueError("Invalid direct-selection scores")
    ranking = sorted(range(n), key=lambda i: (-float(direct_scores[i]), str(ids[i])))
    selected = np.zeros(n)
    selected[ranking[:top_k]] = 1 / top_k
    result["direct_zipfit"] = selected
    # source-by-length strata are supplied as a separate integer domain array
    # below by preparation; this default keeps domain identity fixed.
    diagnostics = {"negative_scores": int((scores < 0).sum()), "zero_score_fallback": bool(score.sum() == 0),
                   "compel_eligible": int(band.sum()), "direct_eligible": top_k}
    return result, diagnostics


def shuffled_bucket_weights(domain, length, bucket, scores) -> np.ndarray:
    """Shuffle document scores, not the order of development views (a no-op)."""
    domain, length, bucket = map(np.asarray, (domain, length, bucket))
    values = np.maximum(np.asarray(scores, dtype=float), 0).copy()
    rng = np.random.default_rng(1405)
    for d in np.unique(domain):
        idx = np.flatnonzero(domain == d)
        boundaries = np.quantile(length[idx], [.25, .5, .75])
        strata = np.searchsorted(boundaries, length[idx], side="right")
        for s in np.unique(strata):
            rows = idx[strata == s]
            values[rows] = rng.permutation(values[rows])
    masses = np.bincount(bucket, weights=values)
    sizes = np.bincount(bucket)
    if masses.sum() <= 0:
        return np.full(len(values), 1 / len(values))
    return masses[bucket] / masses.sum() / sizes[bucket]
