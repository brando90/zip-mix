#!/usr/bin/env python3
"""Sequential, resumable full-finetuning screen. No external model service."""
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
from transformers import AutoModelForCausalLM
import transformers

from common import (CONFIG, METHODS, MODEL, MODEL_REVISION, SEEDS, atomic_json,
                    file_hash, now, stable_hash)


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


def load_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


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
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=inputs["input_ids"].is_cuda):
        output = model(**inputs, **kwargs)
    # Every prompt is LEFT padded, so its actual final token is at -1.
    return output.logits[:, -1, :].float()


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


def new_model(cache, device, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION,
        cache_dir=cache, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        if parameter.dtype != torch.float32:
            raise ValueError("Full fine-tuning requires float32 master parameters")
    return model


def checkpoint_state(model, optimizer, rng, step, tokens, padded_tokens, elapsed, seen, loss_sum):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
            "step": step, "processed_input_tokens": tokens, "padded_input_tokens": padded_tokens, "elapsed_seconds": elapsed,
            "seen": sorted(seen), "loss_sum": loss_sum}


def train_cell(cell, ledger, ledger_path, args, manifest, train, evaluation, weights, device):
    path = args.output / cell["id"]
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = path / "checkpoint.pt"
    prior_status = cell["status"]
    if prior_status == "running":
        resumes = cell.get("resumes", [])
        if len(resumes) >= CONFIG["max_infrastructure_resumes_per_cell"]:
            raise RuntimeError("Prespecified infrastructure resume limit exceeded")
        resumes.append({"at": now(), "previous_progress_step": cell.get("step", 0),
                        "checkpoint_present": checkpoint_path.exists()})
        cell["resumes"] = resumes
    cell.update(status="running", started_at=cell.get("started_at", now()), updated_at=now(), process_id=os.getpid())
    atomic_json(ledger_path, ledger)
    started = time.monotonic()
    model = new_model(args.cache_dir, device, cell["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                 betas=(.9, .95), eps=1e-8, weight_decay=CONFIG["weight_decay"])
    rng = np.random.default_rng(cell["seed"])
    step, tokens, padded_tokens, elapsed, seen, loss_sum = 0, 0, 0, 0.0, set(), 0.0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["sampler_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        step, tokens, elapsed = state["step"], state["processed_input_tokens"], state["elapsed_seconds"]
        padded_tokens = state["padded_input_tokens"]
        seen, loss_sum = set(state["seen"]), state["loss_sum"]
        del state
    if prior_status == "running":
        cell["resumes"][-1].update(checkpoint_step=step,
            replayed_optimizer_steps=max(0, cell.get("step", 0) - step),
            replayed_input_tokens=max(0, cell.get("processed_input_tokens", 0) - tokens))
    prior_elapsed = elapsed
    start_training = time.monotonic()
    # A resume from the final-step checkpoint runs no loop iteration.
    logits = loss = None
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for current in range(step, CONFIG["steps"]):
        indices = rng.choice(len(train["ids"]), CONFIG["batch_size"], replace=True, p=weights[cell["method"]])
        inputs, input_count = left_padded_batch(train, indices, manifest["pad_token_id"], device)
        gold = torch.as_tensor(train["answer_token"][indices], dtype=torch.long, device=device)
        progress = (current + 1 - CONFIG["warmup_steps"]) / max(1, CONFIG["steps"] - CONFIG["warmup_steps"])
        scale = (current + 1) / CONFIG["warmup_steps"] if current < CONFIG["warmup_steps"] else .5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = CONFIG["learning_rate"] * scale
        optimizer.zero_grad(set_to_none=True)
        logits = last_logits(model, inputs)
        # Exactly one supervised next-token position per example. This equals
        # full causal cross entropy with every prompt label masked out.
        loss = torch.nn.functional.cross_entropy(logits, gold)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite gradient norm")
        optimizer.step()
        step = current + 1
        tokens += input_count
        padded_tokens += inputs["input_ids"].numel()
        seen.update(indices.tolist())
        loss_sum += float(loss.detach())
        elapsed = prior_elapsed + time.monotonic() - start_training
        cell.update(step=step, supervised_tokens=step * CONFIG["batch_size"],
                    processed_input_tokens=tokens, padded_input_tokens=padded_tokens, last_training_loss=float(loss.detach()),
                    unique_training_examples=len(seen), elapsed_training_seconds=elapsed, updated_at=now())
        atomic_json(ledger_path, ledger)
        if step % CONFIG["checkpoint_every"] == 0 or step == CONFIG["steps"]:
            atomic_torch(checkpoint_path, checkpoint_state(model, optimizer, rng, step, tokens, padded_tokens, elapsed, seen, loss_sum))
        if step % 16 == 0:
            print(json.dumps({"cell": cell["id"], "step": step, "loss": float(loss.detach()), "elapsed_training_seconds": elapsed}), flush=True)
    torch.cuda.synchronize()
    elapsed = prior_elapsed + time.monotonic() - start_training
    eval_start = time.monotonic()
    metrics = evaluate(model, evaluation, manifest["pad_token_id"], manifest["label_tokens"],
                       args.eval_batch_size, device, path)
    torch.cuda.synchronize()
    model_file = path / "model.pt"
    atomic_torch(model_file, {"state_dict": model.state_dict(), "model": MODEL,
                             "revision": MODEL_REVISION, "cell": cell["id"], "run_id": ledger["run_id"]})
    receipt = {"cell": cell["id"], "method": cell["method"], "seed": cell["seed"],
               "run_id": ledger["run_id"], "manifest_id": manifest["manifest_id"],
               "completed_at": now(), "steps": step, "supervised_tokens": step * CONFIG["batch_size"],
               "processed_input_tokens": tokens, "padded_input_tokens": padded_tokens, "unique_training_examples": len(seen),
               "mean_training_loss": loss_sum / CONFIG["steps"], "training_seconds": elapsed,
               "evaluation_seconds": time.monotonic() - eval_start,
               "this_process_cell_seconds": time.monotonic() - started,
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
               "metrics": metrics, "infrastructure_resumes": cell.get("resumes", []),
               "model_file": "model.pt", "model_sha256": file_hash(model_file)}
    atomic_json(path / "result.json", receipt)
    cell.update(status="complete", completed_at=now(), metrics=metrics,
                result_file=str((path / "result.json").relative_to(args.output)))
    atomic_json(ledger_path, ledger)
    # Final weights, predictions and receipt are durable; optimizer checkpoint
    # is no longer needed for this completed logical cell.
    checkpoint_path.unlink(missing_ok=True)
    del model, optimizer, logits, loss
    gc.collect()
    torch.cuda.empty_cache()


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("This frozen run requires one available CUDA device with bfloat16 support")
    if torch.cuda.device_count() != 1:
        raise SystemExit("Bind exactly one device with CUDA_VISIBLE_DEVICES before launch")
    manifest = json.loads((args.data / "manifest.json").read_text())
    manifest_id = manifest.pop("manifest_id")
    if stable_hash(manifest) != manifest_id:
        raise ValueError("Manifest identity mismatch")
    manifest["manifest_id"] = manifest_id
    if manifest["config"] != CONFIG or manifest["model"] != MODEL or manifest["model_revision"] != MODEL_REVISION:
        raise ValueError("Prepared data and training configuration differ")
    for name, expected in manifest["files"].items():
        if file_hash(args.data / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    code = {name: file_hash(Path(__file__).parent / name) for name in ("common.py", "train_sft.py", "analyze_sft.py")}
    identity = {"manifest_id": manifest_id, "code": code, "config": CONFIG,
                "model": MODEL, "model_revision": MODEL_REVISION, "eval_batch_size": args.eval_batch_size}
    run_id = stable_hash(identity)
    ledger_path = args.output / "ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger["run_id"] != run_id:
            raise ValueError("Run inputs or code changed; create a separate prospective run")
    else:
        cells = [{"id": f"{method}__seed{seed}", "method": method, "seed": seed,
                  "status": "pending" if manifest["method_status"][method] == "available" else "failed",
                  "availability": manifest["method_status"][method],
                  "failure_kind": None if manifest["method_status"][method] == "available" else "scientific_method_unavailable"}
                 for method in METHODS for seed in SEEDS]
        ledger = {"run_id": run_id, "identity": identity, "created_at": now(),
                  "expected_training_cells": len(cells), "base": {"status": "pending"}, "cells": cells,
                  "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                              "visible_devices": 1, "forward_dtype": "bfloat16_autocast", "master_parameter_dtype": "float32", "full_finetuning": True}}
        atomic_json(ledger_path, ledger)
    train = load_arrays(args.data / "train.npz")
    evaluation = {name: load_arrays(args.data / f"{name}.npz") for name in ("target", "retention")}
    selection = load_arrays(args.data / "selection.npz")
    weights = {m: np.asarray(selection[m], dtype=np.float64) for m in METHODS if manifest["method_status"][m] == "available"}
    for method, value in weights.items():
        if len(value) != len(train["ids"]) or not np.isfinite(value).all() or (value < 0).any() or not np.isclose(value.sum(), 1):
            raise ValueError(f"Invalid frozen weights: {method}")
        value /= value.sum()
    device = torch.device("cuda:0")
    if ledger["base"]["status"] not in ("complete", "failed"):
        path = args.output / "base"
        path.mkdir(exist_ok=True)
        ledger["base"].update(status="running", started_at=now())
        atomic_json(ledger_path, ledger)
        start = time.monotonic()
        try:
            model = new_model(args.cache_dir, device, 0)
            metrics = evaluate(model, evaluation, manifest["pad_token_id"], manifest["label_tokens"],
                               args.eval_batch_size, device, path)
            receipt = {"run_id": run_id, "model": MODEL, "model_revision": MODEL_REVISION,
                       "metrics": metrics, "elapsed_seconds": time.monotonic() - start,
                       "completed_at": now()}
            atomic_json(path / "result.json", receipt)
            ledger["base"].update(status="complete", completed_at=now(), metrics=metrics)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            ledger["base"].update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_at=now())
            atomic_json(path / "failure.json", {"error": ledger["base"]["error"], "traceback": traceback.format_exc()})
        atomic_json(ledger_path, ledger)
    for cell in ledger["cells"]:
        if cell["status"] in ("complete", "failed", "unavailable"):
            continue
        try:
            train_cell(cell, ledger, ledger_path, args, manifest, train, evaluation, weights, device)
        except Exception as exc:
            cell.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_at=now())
            path = args.output / cell["id"]
            path.mkdir(exist_ok=True)
            atomic_json(path / "failure.json", {"error": cell["error"], "traceback": traceback.format_exc(), "cell": cell})
            atomic_json(ledger_path, ledger)
            print(json.dumps({"cell": cell["id"], "status": "failed", "error": cell["error"]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    ledger["updated_at"] = now()
    ledger["status"] = "complete" if all(c["status"] == "complete" for c in ledger["cells"]) and ledger["base"]["status"] == "complete" else "partial_with_failures"
    atomic_json(ledger_path, ledger)
    print(json.dumps({"status": ledger["status"], "complete": sum(c["status"] == "complete" for c in ledger["cells"]),
                      "expected": len(ledger["cells"])}), flush=True)
    return ledger["status"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.eval_batch_size < 1:
        raise SystemExit("Evaluation batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(args.output / ".owner.lock"):
        status = run(args)
    # Nonzero unless every cell and the base evaluation completed, so an
    # external monitor cannot read a partial ledger as procedure completion.
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
