#!/usr/bin/env python3
"""Sequential, resumable full fine-tuning for both Experiment 07 stages.

Stage A (calibration data): learning-rate grid x seeds, uniform sampling.
Stage B (stage_b data): three frozen selectors x seeds at the learning rate a
passed Stage A decision selected. Training, evaluation and padding follow
Experiment 05 v2's train_sft.py; differences are listed in PROTOCOL.md.

--smoke runs a tiny randomly initialized Qwen2 on CPU for a few steps; it
checks the plumbing only and writes a ledger marked smoke.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import inspect
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from calib_common import (HOME, MODEL, MODEL_REVISION, STAGE_A, STAGE_B, TRAIN, atomic_json,
                          file_hash, now, stable_hash)

SMOKE = {"steps": 4, "batch_size": 4, "evaluation_items": 32, "checkpoint_every": 2}


@contextmanager
def exclusive_run(path):
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this output directory") from exc
        yield


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_arrays(path, limit=None):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key][:limit] if limit else arrays[key] for key in arrays.files}


def left_padded_batch(data, indices, pad_token, device):
    lengths = data["lengths"][indices]
    width = int(lengths.max())
    tokens = np.full((len(indices), width), pad_token, dtype=np.int64)
    mask = np.zeros_like(tokens)
    for i, (row, length) in enumerate(zip(indices, lengths)):
        tokens[i, -int(length):] = data["input_ids"][row, :length]
        mask[i, -int(length):] = 1
    input_ids = torch.as_tensor(tokens, device=device)
    attention_mask = torch.as_tensor(mask, device=device)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "position_ids": position_ids}, int(lengths.sum())


def last_logits(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = {"use_cache": False}
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        kwargs["num_logits_to_keep"] = 1
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=inputs["input_ids"].is_cuda):
        output = model(**inputs, **kwargs)
    # Every prompt is LEFT padded, so its actual final token is at -1.
    return output.logits[:, -1, :].float()


def learning_rate_scale(step_index, steps, warmup):
    """Linear warmup then cosine decay to zero; identical to Experiment 05."""
    if step_index < warmup:
        return (step_index + 1) / warmup
    progress = (step_index + 1 - warmup) / max(1, steps - warmup)
    return .5 * (1 + math.cos(math.pi * progress))


def evaluate(model, datasets, pad_token, labels, batch_size, device, output_dir):
    model.eval()
    results = {}
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    for name, data in datasets.items():
        correct, predictions, nll_values = [], [], []
        for start in range(0, len(data["ids"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(data["ids"])))
            inputs, _ = left_padded_batch(data, indices, pad_token, device)
            with torch.inference_mode():
                logits = last_logits(model, inputs)
                scores = logits.index_select(-1, label_tensor)
                valid = torch.arange(len(labels), device=device)[None, :] < torch.as_tensor(data["nchoices"][indices], device=device)[:, None]
                scores = scores.masked_fill(~valid, -torch.inf)
                predicted = scores.argmax(-1).cpu().numpy()
                gold = torch.as_tensor(data["answer_token"][indices], dtype=torch.long, device=device)
                losses = torch.nn.functional.cross_entropy(logits, gold, reduction="none")
            if not torch.isfinite(losses).all():
                raise ValueError("Nonfinite evaluation loss")
            predictions.extend(predicted.tolist())
            correct.extend((predicted == data["answer"][indices]).astype(int).tolist())
            nll_values.extend(losses.cpu().tolist())
        np.savez_compressed(output_dir / f"{name}_predictions.npz", ids=data["ids"],
                            correct=np.asarray(correct), prediction=np.asarray(predictions),
                            gold=data["answer"], label_nll=np.asarray(nll_values))
        results[name] = {"correct": int(sum(correct)), "count": len(correct),
                         "accuracy": float(np.mean(correct)), "label_nll": float(np.mean(nll_values))}
    return results


def new_model(args, device, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.smoke:
        config = Qwen2Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
                             tie_word_embeddings=True)
        model = Qwen2ForCausalLM(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, cache_dir=args.cache_dir,
                                                     torch_dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        if parameter.dtype != torch.float32:
            raise ValueError("Full fine-tuning requires float32 master parameters")
    return model


def rng_state(device):
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def plan_cells(manifest, decision, smoke):
    """The frozen denominator: every declared cell, in fixed order."""
    if manifest["stage"] == "calibration":
        rates, seeds = STAGE_A["learning_rates"], STAGE_A["seeds"]
        if smoke:
            rates, seeds = rates[:2], seeds[:1]
        cells = [{"id": f"lr{lr:.0e}__seed{seed}", "learning_rate": lr, "method": "uniform", "seed": seed}
                 for lr in rates for seed in seeds]
    else:
        lr = decision["selected_learning_rate"]
        if lr is None and smoke:
            lr = STAGE_A["reference_learning_rate"]
        seeds = STAGE_B["seeds"][:1] if smoke else STAGE_B["seeds"]
        cells = [{"id": f"{method}__seed{seed}", "learning_rate": lr, "method": method, "seed": seed}
                 for method in STAGE_B["methods"] for seed in seeds]
    return [{**c, "status": "pending"} for c in cells]


def train_cell(cell, ledger, ledger_path, args, manifest, train, evaluation, weights, device, steps, batch, every):
    path = args.output / cell["id"]
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = path / "checkpoint.pt"
    if cell["status"] == "running":
        resumes = cell.setdefault("resumes", [])
        if len(resumes) >= TRAIN["max_infrastructure_resumes_per_cell"]:
            raise RuntimeError("Prespecified infrastructure resume limit exceeded")
        resumes.append({"at": now(), "previous_progress_step": cell.get("step", 0),
                        "checkpoint_present": checkpoint_path.exists()})
    cell.update(status="running", started_at=cell.get("started_at", now()), updated_at=now(), process_id=os.getpid())
    atomic_json(ledger_path, ledger)
    started = time.monotonic()
    model = new_model(args, device, cell["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cell["learning_rate"], betas=tuple(TRAIN["betas"]),
                                  eps=TRAIN["eps"], weight_decay=TRAIN["weight_decay"])
    rng = np.random.default_rng(cell["seed"])
    step, tokens, padded, elapsed, seen, loss_sum = 0, 0, 0, 0.0, set(), 0.0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["sampler_rng"]
        restore_rng(state["rng"])
        step, tokens, padded, elapsed = state["step"], state["tokens"], state["padded"], state["elapsed"]
        seen, loss_sum = set(state["seen"]), state["loss_sum"]
        del state
    if cell.get("resumes"):
        cell["resumes"][-1].update(checkpoint_step=step, replayed_optimizer_steps=max(0, cell.get("step", 0) - step))
    prior_elapsed, start_training = elapsed, time.monotonic()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    p = weights[cell["method"]]
    for current in range(step, steps):
        indices = rng.choice(len(train["ids"]), batch, replace=True, p=p)
        inputs, count = left_padded_batch(train, indices, manifest["pad_token_id"], device)
        gold = torch.as_tensor(train["answer_token"][indices], dtype=torch.long, device=device)
        for group in optimizer.param_groups:
            group["lr"] = cell["learning_rate"] * learning_rate_scale(current, steps, TRAIN["warmup_steps"] if not args.smoke else 1)
        optimizer.zero_grad(set_to_none=True)
        # Exactly one supervised next-token position per example.
        loss = torch.nn.functional.cross_entropy(last_logits(model, inputs), gold)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["gradient_clip"])
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite gradient norm")
        optimizer.step()
        step = current + 1
        tokens += count
        padded += inputs["input_ids"].numel()
        seen.update(indices.tolist())
        loss_sum += float(loss.detach())
        elapsed = prior_elapsed + time.monotonic() - start_training
        cell.update(step=step, processed_input_tokens=tokens, padded_input_tokens=padded,
                    last_training_loss=float(loss.detach()), elapsed_training_seconds=elapsed, updated_at=now())
        atomic_json(ledger_path, ledger)
        if step % every == 0 and step < steps:
            atomic_torch(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "sampler_rng": rng.bit_generator.state, "rng": rng_state(device), "step": step, "tokens": tokens,
                "padded": padded, "elapsed": elapsed, "seen": sorted(seen), "loss_sum": loss_sum})
        if step % 16 == 0 or args.smoke:
            print(json.dumps({"cell": cell["id"], "step": step, "loss": float(loss.detach()), "elapsed": round(elapsed, 2)}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_start = time.monotonic()
    metrics = evaluate(model, evaluation, manifest["pad_token_id"], manifest["label_tokens"], args.eval_batch_size, device, path)
    receipt = {"cell": cell["id"], "method": cell["method"], "seed": cell["seed"], "learning_rate": cell["learning_rate"],
               "run_id": ledger["run_id"], "manifest_id": manifest["manifest_id"], "completed_at": now(),
               "steps": step, "supervised_tokens": step * batch, "processed_input_tokens": tokens,
               "padded_input_tokens": padded, "unique_training_examples": len(seen),
               "mean_training_loss": loss_sum / max(1, steps), "training_seconds": elapsed,
               "evaluation_seconds": time.monotonic() - eval_start, "this_process_cell_seconds": time.monotonic() - started,
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
               "metrics": metrics, "infrastructure_resumes": cell.get("resumes", [])}
    if args.save_models:
        atomic_torch(path / "model.pt", {"state_dict": model.state_dict(), "model": MODEL, "revision": MODEL_REVISION,
                                         "cell": cell["id"], "run_id": ledger["run_id"]})
        receipt["model_sha256"] = file_hash(path / "model.pt")
    atomic_json(path / "result.json", receipt)
    cell.update(status="complete", completed_at=now(), metrics=metrics, result_file=f"{cell['id']}/result.json")
    atomic_json(ledger_path, ledger)
    checkpoint_path.unlink(missing_ok=True)
    del model, optimizer, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def verified_manifest(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest_id = manifest.pop("manifest_id")
    if stable_hash(manifest) != manifest_id:
        raise ValueError("Manifest identity mismatch")
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    manifest["manifest_id"] = manifest_id
    if manifest["model"] != MODEL or manifest["model_revision"] != MODEL_REVISION or manifest["train_config"] != TRAIN:
        raise ValueError("Prepared data and training configuration differ")
    return manifest


def verified_decision(path, smoke):
    """Stage B uses only a passed, hash-verified Stage A decision."""
    decision = json.loads(Path(path).read_text())
    decision_id = decision.pop("decision_id")
    if stable_hash(decision) != decision_id:
        raise ValueError("Stage A decision identity mismatch")
    if not decision["gate"]["passed"] and not smoke:
        raise SystemExit("Stage A gate did not pass; Stage B is not admitted (PROTOCOL.md)")
    if decision.get("smoke") and not smoke:
        raise SystemExit("A smoke decision cannot admit a real Stage B run")
    return {**decision, "decision_id": decision_id}


def run(args):
    manifest = verified_manifest(args.data)
    if args.smoke:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise SystemExit("This frozen run requires one CUDA device with bfloat16 support")
        if torch.cuda.device_count() != 1:
            raise SystemExit("Bind exactly one device with CUDA_VISIBLE_DEVICES before launch")
        device = torch.device("cuda:0")
    decision = None
    if manifest["stage"] == "stage_b":
        if not args.decision:
            raise SystemExit("Stage B requires --decision from analyze_calibration.py --stage calibration")
        decision = verified_decision(args.decision, args.smoke)
    steps = SMOKE["steps"] if args.smoke else TRAIN["steps"]
    batch = SMOKE["batch_size"] if args.smoke else TRAIN["batch_size"]
    every = SMOKE["checkpoint_every"] if args.smoke else TRAIN["checkpoint_every"]
    limit = SMOKE["evaluation_items"] if args.smoke else None
    code = {n: file_hash(HOME / n) for n in ("calib_common.py", "train_calibration.py", "analyze_calibration.py")}
    identity = {"manifest_id": manifest["manifest_id"], "code": code, "train": TRAIN, "model": MODEL,
                "model_revision": MODEL_REVISION, "eval_batch_size": args.eval_batch_size, "smoke": args.smoke,
                "decision_id": decision["decision_id"] if decision else None}
    run_id = stable_hash(identity)
    ledger_path = args.output / "ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger["run_id"] != run_id:
            raise ValueError("Run inputs or code changed; create a separate prospective run")
    else:
        cells = plan_cells(manifest, decision, args.smoke)
        ledger = {"run_id": run_id, "identity": identity, "stage": manifest["stage"], "created_at": now(),
                  "smoke": args.smoke, "expected_training_cells": len(cells), "base": {"status": "pending"},
                  "cells": cells, "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                  "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}}
        atomic_json(ledger_path, ledger)
    train = load_arrays(args.data / "train.npz")
    evaluation = {n: load_arrays(args.data / f"{n}.npz", limit) for n in manifest["evaluation_sets"]}
    if manifest["stage"] == "calibration":
        weights = {"uniform": np.full(len(train["ids"]), 1 / len(train["ids"]))}
    else:
        selection = load_arrays(args.data / "selection.npz")
        weights = {m: np.asarray(selection[m], dtype=np.float64) for m in STAGE_B["methods"]}
    for method, value in weights.items():
        if len(value) != len(train["ids"]) or not np.isfinite(value).all() or (value <= 0).any() or not np.isclose(value.sum(), 1):
            raise ValueError(f"Invalid frozen weights: {method}")
        value /= value.sum()
    if ledger["base"]["status"] not in ("complete", "failed"):
        path = args.output / "base"
        path.mkdir(exist_ok=True)
        start = time.monotonic()
        try:
            model = new_model(args, device, 0)
            metrics = evaluate(model, evaluation, manifest["pad_token_id"], manifest["label_tokens"], args.eval_batch_size, device, path)
            atomic_json(path / "result.json", {"run_id": run_id, "model": MODEL, "model_revision": MODEL_REVISION,
                        "metrics": metrics, "elapsed_seconds": time.monotonic() - start, "completed_at": now()})
            ledger["base"].update(status="complete", completed_at=now(), metrics=metrics)
            del model
        except Exception as exc:
            ledger["base"].update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_at=now())
            atomic_json(path / "failure.json", {"error": ledger["base"]["error"], "traceback": traceback.format_exc()})
        atomic_json(ledger_path, ledger)
    for cell in ledger["cells"]:
        if cell["status"] in ("complete", "failed"):
            continue
        try:
            train_cell(cell, ledger, ledger_path, args, manifest, train, evaluation, weights, device, steps, batch, every)
        except Exception as exc:
            cell.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_at=now())
            path = args.output / cell["id"]
            path.mkdir(exist_ok=True)
            atomic_json(path / "failure.json", {"error": cell["error"], "traceback": traceback.format_exc(), "cell": cell})
            atomic_json(ledger_path, ledger)
            print(json.dumps({"cell": cell["id"], "status": "failed", "error": cell["error"]}), flush=True)
    complete = all(c["status"] == "complete" for c in ledger["cells"]) and ledger["base"]["status"] == "complete"
    ledger.update(updated_at=now(), status="complete" if complete else "partial_with_failures")
    atomic_json(ledger_path, ledger)
    print(json.dumps({"status": ledger["status"], "complete": sum(c["status"] == "complete" for c in ledger["cells"]),
                      "expected": len(ledger["cells"])}), flush=True)
    return ledger["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="data/calibration or data/stage_b")
    parser.add_argument("--output", type=Path, required=True, help="new private run directory")
    parser.add_argument("--decision", type=Path, help="Stage A decision.json (Stage B only)")
    parser.add_argument("--cache-dir")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--save-models", action="store_true", help="also keep final weights (~2 GB per cell)")
    parser.add_argument("--smoke", action="store_true", help="tiny random model on CPU; plumbing only")
    args = parser.parse_args()
    if args.eval_batch_size < 1:
        raise SystemExit("Evaluation batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(args.output / ".owner.lock"):
        status = run(args)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
