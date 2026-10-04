#!/usr/bin/env python3
"""Compact, from-scratch mixture comparison; not paper-scale reproductions.

DoReMi: https://arxiv.org/abs/2305.10429, Algorithm 1; token-count reference,
clipped token excess for weights, unclipped weighted NLL for proxy gradients.
DoGE: https://proceedings.mlr.press/v235/fan24e.html, eqs 4-5; full-parameter
train/development gradient inner products. As in its official trainer, each
vector is norm-clipped and alignment is divided by mean train-gradient norm.
One pooled held-out development target replaces the paper's target-domain sum.
Both use balanced per-domain proxy minibatches and transfer their time-average
weights to a fresh SAME-SIZE final model. This is a declared small-scale
adaptation, not cross-scale transfer. Final training tokens match across arms;
reference, proxy, development-gradient and evaluation costs are additional.

No evaluation array is read until a cell's final model is fully trained.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import tempfile
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("token_proportional", "uniform_domain", "uniform_bucket", "compel_filter",
        "zipmix_static", "direct_zipfit", "shuffled_zipmix", "doremi", "doge")


@dataclass
class Config:
    # All defaults are recorded verbatim in frozen_config.json before training.
    arms: list[str] = field(default_factory=lambda: list(ARMS))
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    vocab_size: int = 50257
    seq_len: int = 128
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    batch_size: int = 16
    final_steps: int = 1000
    reference_steps: int = 1000
    proxy_steps: int = 1000
    proxy_per_domain_batch: int = 2
    dev_batch_size: int = 16
    learning_rate: float = 0.0005
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    tau: float = 0.001
    doremi_eta: float = 1.0
    smoothing: float = 0.001
    doge_mu: float = 0.01
    doge_smoothing: float = 0.0
    doge_score_clip: float = 5.0
    compel_min_cr: float = 0.65
    compel_max_cr: float = 0.80
    checkpoint_every: int = 100
    log_every: int = 25
    eval_batch_size: int = 16
    device: str = "cuda"
    precision: str = "bf16"
    num_threads: int = 4
    sampling_seed_offset: int = 10000
    max_recoveries: int = 1
    metadata: dict = field(default_factory=dict)

    def validate(self):
        for name in ("vocab_size", "seq_len", "d_model", "n_layers", "n_heads", "d_ff",
                     "batch_size", "final_steps", "reference_steps", "proxy_steps",
                     "proxy_per_domain_batch", "dev_batch_size", "checkpoint_every",
                     "eval_batch_size", "num_threads", "log_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide into n_heads")
        if len(set(self.arms)) != len(self.arms) or not set(self.arms) <= set(ARMS):
            raise ValueError("arms must be distinct known methods")
        if not self.arms or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("nonempty unique arms and seeds required")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("only fp32 and bf16 supported; no unscaled fp16")
        if not 0 <= self.dropout < 1 or not 0 <= self.smoothing < 1 or not 0 <= self.doge_smoothing < 1:
            raise ValueError("dropout and smoothing must be in [0,1)")
        if self.tau <= 0 or self.learning_rate <= 0 or self.doge_mu <= 0 or self.grad_clip <= 0:
            raise ValueError("tau, learning_rate, doge_mu and grad_clip must be positive")
        if self.doge_score_clip <= 0 or self.doremi_eta <= 0 or self.warmup_steps < 0:
            raise ValueError("invalid mixture or warmup settings")
        if not 0 <= self.min_lr_ratio <= 1 or self.max_recoveries < 0:
            raise ValueError("invalid min_lr_ratio or max_recoveries")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, obj):
    atomic_write(Path(path), lambda f: f.write((json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or np.any(x < 0) or not np.isfinite(x).all() or x.sum() <= 0:
        raise ValueError("probability mass must be finite, nonnegative, and nonzero")
    return x / x.sum()


def grouped_probabilities(groups, mass):
    """Map group masses to uniform sampling within each observed group."""
    ids, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(mass) != len(ids):
        raise ValueError("mass must correspond to the sorted observed groups only")
    return normalize(mass)[inverse] / counts[inverse]


def sampling_probabilities(arm, data, cfg, seed):
    n = len(data["tokens"])
    domain, bucket = data["domain"], data["bucket"]
    score = np.maximum(np.asarray(data["score"], dtype=np.float64), 0)
    if arm == "token_proportional":
        return np.full(n, 1 / n)  # Every chunk has the same number of loss tokens.
    if arm in ("uniform_domain", "uniform_bucket"):
        groups = domain if arm == "uniform_domain" else bucket
        return grouped_probabilities(groups, np.ones(len(np.unique(groups))))
    if arm == "compel_filter":
        keep = (data["cr"] >= cfg.compel_min_cr) & (data["cr"] <= cfg.compel_max_cr)
        if not keep.any():
            raise ValueError("Compel band contains no chunks; do not silently substitute a baseline")
        return normalize(keep)
    if arm == "direct_zipfit":
        if "mean_score" not in data:
            raise ValueError("direct_zipfit requires paper-style mean_score; max score cannot substitute")
        # Continuous mean-score sampling is explicitly a soft ZIP-FIT adaptation.
        return normalize(np.maximum(data["mean_score"], 0) + cfg.tau)
    if arm in ("zipmix_static", "shuffled_zipmix"):
        if arm == "shuffled_zipmix":
            score = np.random.default_rng(seed + cfg.sampling_seed_offset + 97).permutation(score)
        ids = np.unique(bucket)
        mass = [score[bucket == k].sum() + cfg.tau for k in ids]
        return grouped_probabilities(bucket, mass)
    raise ValueError(f"{arm} requires a learned mixture")


def exponentiated_update(weights, scores, eta=1.0, smoothing=0.0):
    w = torch.as_tensor(weights, dtype=torch.float64)
    s = torch.as_tensor(scores, dtype=torch.float64, device=w.device)
    if not torch.isfinite(w).all() or not torch.isfinite(s).all() or torch.any(w <= 0):
        raise ValueError("nonfinite score or nonpositive weight in mixture update")
    p = torch.softmax(w.log() + eta * s, dim=0)
    return (1 - smoothing) * p + smoothing / len(p)


def clipped_excess(token_nll, reference_nll):
    """Clip each token before averaging; clipping a domain mean is different."""
    return (token_nll - reference_nll).clamp_min(0).mean()


def gradient_vector(loss, params):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params)])


def clip_vector(vector, max_norm):
    return vector * torch.clamp(max_norm / (vector.norm() + 1e-6), max=1.0)


def doge_update(weights, train_gradients, target_gradient, lr, mu, smoothing=0.0, score_clip=5.0):
    """All-parameter inner products, normalized as official DoGE trainer.

    Inputs are independently norm-clipped gradient vectors. Scores divided by
    mean train norm preserve magnitudes (this is not cosine similarity).
    lr/mu scales the mirror-descent step. Single held-out target uses no diagonal
    subtraction: target examples are separate from every training domain batch.
    """
    scores = train_gradients @ target_gradient
    scores = scores / (train_gradients.norm(dim=1).mean() + 1e-6)
    scores = scores.clamp(-score_clip, score_clip)
    new_weights = exponentiated_update(weights, scores, lr / mu, smoothing)
    return new_weights, scores


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dropout = cfg.n_heads, cfg.dropout
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [a.reshape(b, t, self.heads, c // self.heads).transpose(1, 2) for a in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                         dropout_p=self.dropout if self.training else 0.0)
        x = x + F.dropout(self.proj(y.transpose(1, 2).reshape(b, t, c)), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.ln2(x)), self.dropout, self.training)


class CausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.mlp[-1].weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return F.linear(self.ln(x), self.token.weight)


def causal_token_nll(model, tokens):
    """tokens has L+1 entries: input tokens[:-1], targets tokens[1:]."""
    logits = model(tokens[:, :-1])
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           tokens[:, 1:].reshape(-1), reduction="none").reshape(tokens.shape[0], -1)


def autocast_context(cfg):
    if cfg.precision == "bf16":
        return torch.autocast(device_type=torch.device(cfg.device).type, dtype=torch.bfloat16)
    return nullcontext()


def sync(cfg):
    if torch.device(cfg.device).type == "cuda":
        torch.cuda.synchronize()


def batch_tensor(tokens, indices, cfg):
    return torch.as_tensor(tokens[indices].astype(np.int64), device=cfg.device)


def lr_for_step(step, total, cfg):
    warmup = min(cfg.warmup_steps, max(total - 1, 0))
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    frac = (step - warmup) / max(total - warmup - 1, 1)
    return cfg.learning_rate * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * frac)) / 2)


def rng_state(rng):
    return {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, rng):
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def failure_kind(error):
    """Numerical failures are never retried; only infrastructure faults may resume."""
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    text = str(error).lower()
    if isinstance(error, FloatingPointError) or "nonfinite" in text or "non-finite" in text:
        return "numerical"
    return "runtime"


def load_pt(path):
    # Only locally authored, fingerprint-bound checkpoints; never external pickles.
    return torch.load(path, map_location="cpu", weights_only=False)


def train_stage(cfg, train, dev, directory, stage, seed, fingerprint,
                probabilities=None, reference=None, method=None):
    """Resume exactly from last durable step, never from an evaluated checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    if torch.device(cfg.device).type == "cuda":
        # Stage peak includes any resident reference model used by this stage.
        torch.cuda.reset_peak_memory_stats()
    total = getattr(cfg, f"{stage}_steps")
    stage_seed = seed + {"final": 0, "reference": 1000000, "proxy": 2000000}[stage]
    torch.manual_seed(stage_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stage_seed)
    model = CausalLM(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(stage_seed + cfg.sampling_seed_offset)
    params = list(model.parameters())
    nparams = sum(p.numel() for p in params)
    domain_ids = np.unique(train["domain"])
    domain_indices = [np.flatnonzero(train["domain"] == k) for k in domain_ids]
    k = len(domain_ids)
    weights = torch.full((k,), 1 / k, dtype=torch.float64, device=cfg.device)
    weight_sum = torch.zeros_like(weights)
    seen = np.zeros(len(train["tokens"]), dtype=np.int64)
    counters = {"optimizer_tokens": 0, "forward_only_tokens": 0, "gradient_probe_tokens": 0}
    start_step, elapsed_previous = 0, 0.0
    checkpoint = directory / f"{stage}_checkpoint.pt"
    if checkpoint.exists():
        saved = load_pt(checkpoint)
        if saved["fingerprint"] != fingerprint or saved["stage"] != stage:
            raise ValueError("checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        weights = saved["weights"].to(cfg.device)
        weight_sum = saved["weight_sum"].to(cfg.device)
        counters, seen = saved["counters"], saved["seen"]
        start_step, elapsed_previous = saved["step"], saved["wall_seconds"]
        restore_rng(saved["rng"], rng)
    if reference is not None:
        reference.eval()
        reference.requires_grad_(False)
    sync(cfg)
    started = time.monotonic()
    last_loss = None

    def save(step):
        sync(cfg)
        wall = elapsed_previous + time.monotonic() - started
        payload = {"fingerprint": fingerprint, "stage": stage, "step": step,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "weights": weights.detach().cpu(), "weight_sum": weight_sum.detach().cpu(),
                   "rng": rng_state(rng), "seen": seen, "counters": counters,
                   "wall_seconds": wall}
        atomic_write(checkpoint, lambda f: torch.save(payload, f))
        atomic_json(directory / f"{stage}_progress.json", {
            "stage": stage, "step": step, "total_steps": total, "last_train_nll": last_loss,
            "wall_seconds": wall, "updated_at": utc_now(), **counters})

    model.train()
    for step in range(start_step, total):
        lr = lr_for_step(step, total, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        if stage != "proxy":
            indices = rng.choice(len(train["tokens"]), cfg.batch_size, replace=True, p=probabilities)
            with autocast_context(cfg):
                loss = causal_token_nll(model, batch_tensor(train["tokens"], indices, cfg)).mean()
            loss.backward()
            counters["optimizer_tokens"] += cfg.batch_size * cfg.seq_len
        else:
            indices_by_domain = [rng.choice(v, cfg.proxy_per_domain_batch, replace=True) for v in domain_indices]
            indices = np.concatenate(indices_by_domain)
            if method == "doremi":
                with autocast_context(cfg):
                    nll = causal_token_nll(model, batch_tensor(train["tokens"], indices, cfg))
                    with torch.no_grad():
                        ref_nll = causal_token_nll(reference, batch_tensor(train["tokens"], indices, cfg))
                scores = (nll.detach() - ref_nll).clamp_min(0).reshape(k, -1).mean(dim=1)
                weights = exponentiated_update(weights, scores, cfg.doremi_eta, cfg.smoothing)
                # Reference terms have zero derivative; clipping is only for weight updates.
                loss = (nll.reshape(k, -1).mean(dim=1) * weights.to(nll.dtype)).sum()
                loss.backward()
                counters["forward_only_tokens"] += len(indices) * cfg.seq_len
            elif method == "doge":
                target_indices = rng.choice(len(dev["tokens"]), cfg.dev_batch_size, replace=True)
                with autocast_context(cfg):
                    dev_loss = causal_token_nll(model, batch_tensor(dev["tokens"], target_indices, cfg)).mean()
                target_gradient = clip_vector(gradient_vector(dev_loss, params), cfg.grad_clip)
                train_gradients, losses = [], []
                for domain_batch in indices_by_domain:
                    with autocast_context(cfg):
                        domain_loss = causal_token_nll(model, batch_tensor(train["tokens"], domain_batch, cfg)).mean()
                    train_gradients.append(clip_vector(gradient_vector(domain_loss, params), cfg.grad_clip))
                    losses.append(domain_loss.detach())
                train_gradients = torch.stack(train_gradients)
                weights, scores = doge_update(weights, train_gradients, target_gradient, lr, cfg.doge_mu,
                                              cfg.doge_smoothing, cfg.doge_score_clip)
                combined_gradient = weights.to(train_gradients.dtype) @ train_gradients
                offset = 0
                for p in params:
                    p.grad = combined_gradient[offset:offset + p.numel()].reshape_as(p).clone()
                    offset += p.numel()
                loss = (torch.stack(losses) * weights).sum()
                counters["gradient_probe_tokens"] += cfg.dev_batch_size * cfg.seq_len
            else:
                raise ValueError("unknown proxy method")
            weight_sum += weights
            counters["optimizer_tokens"] += len(indices) * cfg.seq_len
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite {stage} training loss at step {step}")
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        np.add.at(seen, indices, 1)
        last_loss = float(loss.detach().cpu())
        if (step + 1) % cfg.log_every == 0 or step + 1 == total:
            event = {"at": utc_now(), "stage": stage, "step": step + 1,
                     "train_nll": last_loss, "learning_rate": lr}
            if stage == "proxy":
                event.update(weights=weights.tolist(), scores=scores.tolist())
            with open(directory / "training.jsonl", "a") as f:
                f.write(json.dumps(event, allow_nan=False) + "\n")
                f.flush()
            print(json.dumps({"cell": directory.name, **event}), flush=True)
        if (step + 1) % cfg.checkpoint_every == 0 or step + 1 == total:
            save(step + 1)
    sync(cfg)
    elapsed = elapsed_previous + time.monotonic() - started
    stats = {"stage": stage, "steps": total, "parameters": nparams, "wall_seconds": elapsed,
             **counters, "unique_chunks_seen": int((seen > 0).sum()),
             "chunks_sampled": int(seen.sum()),
             "flops_proxy": int(nparams * (6 * (counters["optimizer_tokens"] + counters["gradient_probe_tokens"])
                                              + 2 * counters["forward_only_tokens"])),
             "flops_definition": "6*P*gradient_tokens + 2*P*forward_only_tokens; excludes attention quadratic, optimizer and preprocessing costs",
             "seen_by_domain": {str(v): int(seen[train["domain"] == v].sum()) for v in domain_ids},
             "seen_by_bucket": {str(v): int(seen[train["bucket"] == v].sum()) for v in np.unique(train["bucket"])}}
    stats["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.device(cfg.device).type == "cuda" else None
    stats["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.device(cfg.device).type == "cuda" else None
    stats["peak_memory_scope"] = "this stage invocation, including resident reference model; earlier recovery invocations not reconstructed"
    if stage == "proxy":
        stats["average_domain_weights"] = (weight_sum / total).tolist()
        stats["domain_ids"] = domain_ids.tolist()
    atomic_json(directory / f"{stage}_stats.json", stats)
    return model, stats


@torch.no_grad()
def evaluate(model, eval_path, directory, cfg):
    """Called only after training. Full fixed evaluation, without early stopping."""
    model.eval()
    sync(cfg)
    started = time.monotonic()
    metrics, arrays = {}, {}
    with np.load(eval_path, allow_pickle=False) as data:
        for split in ("target", "broad"):
            tokens, domains = data[f"{split}_tokens"], data[f"{split}_domain"]
            validate_tokens(tokens, cfg, split)
            if len(tokens) != len(domains):
                raise ValueError("evaluation domain length mismatch")
            losses = []
            for i in range(0, len(tokens), cfg.eval_batch_size):
                with autocast_context(cfg):
                    nll = causal_token_nll(model, batch_tensor(tokens, slice(i, i + cfg.eval_batch_size), cfg))
                losses.append(nll.mean(dim=1).double().cpu().numpy())
            per_example = np.concatenate(losses)
            if not np.isfinite(per_example).all():
                raise ValueError("nonfinite evaluation loss")
            by_domain = {str(k): float(per_example[domains == k].mean()) for k in np.unique(domains)}
            nll = float(per_example.mean())
            metrics[split] = {"nll": nll, "perplexity": math.exp(nll),
                              "domain_macro_nll": float(np.mean(list(by_domain.values()))),
                              "worst_domain_nll": max(by_domain.values()),
                              "domain_nll": by_domain, "examples": len(tokens),
                              "loss_tokens": len(tokens) * cfg.seq_len}
            arrays[f"{split}_nll"] = per_example
            arrays[f"{split}_domain"] = domains
    atomic_write(directory / "heldout_nll.npz", lambda f: np.savez_compressed(f, **arrays))
    sync(cfg)
    metrics["eval_wall_seconds"] = time.monotonic() - started
    return metrics


def validate_tokens(tokens, cfg, name):
    if tokens.ndim != 2 or len(tokens) == 0 or tokens.shape[1] != cfg.seq_len + 1:
        raise ValueError(f"{name} tokens must be nonempty [N, seq_len+1]")
    if not np.issubdtype(tokens.dtype, np.integer) or tokens.min() < 0 or tokens.max() >= cfg.vocab_size:
        raise ValueError(f"{name} contains invalid token ids")


def load_training_data(data_dir, cfg):
    with np.load(data_dir / "train.npz", allow_pickle=False) as source:
        train = {key: source[key] for key in source.files}
    with np.load(data_dir / "dev.npz", allow_pickle=False) as source:
        dev = {key: source[key] for key in source.files}
    validate_tokens(train["tokens"], cfg, "train")
    validate_tokens(dev["tokens"], cfg, "development")
    for key in ("domain", "bucket", "score", "cr"):
        if train[key].shape != (len(train["tokens"]),) or not np.isfinite(train[key]).all():
            raise ValueError(f"train {key} missing, wrong shape or nonfinite")
    for key in ("domain", "bucket"):
        if not np.issubdtype(train[key].dtype, np.integer) or (train[key] < 0).any():
            raise ValueError(f"train {key} must be nonnegative integer labels")
    if "mean_score" in train and (train["mean_score"].shape != train["score"].shape or not np.isfinite(train["mean_score"]).all()):
        raise ValueError("invalid mean_score")
    if dev["domain"].shape != (len(dev["tokens"]),):
        raise ValueError("development domains mismatch")
    return train, dev


def run_cell(arm, seed, cfg, train, dev, eval_path, directory, fingerprint):
    started = time.monotonic()
    stages = {}
    if arm in ("doremi", "doge"):
        reference = None
        if arm == "doremi":
            reference, stages["reference"] = train_stage(
                cfg, train, dev, directory, "reference", seed, fingerprint,
                probabilities=sampling_probabilities("token_proportional", train, cfg, seed))
        proxy, stages["proxy"] = train_stage(cfg, train, dev, directory, "proxy", seed, fingerprint,
                                              reference=reference, method=arm)
        probabilities = grouped_probabilities(train["domain"], stages["proxy"]["average_domain_weights"])
        del proxy, reference
        if torch.device(cfg.device).type == "cuda":
            torch.cuda.empty_cache()
    else:
        probabilities = sampling_probabilities(arm, train, cfg, seed)
    atomic_json(directory / "mixture.json", {
        "arm": arm, "seed": seed, "nonzero_chunks": int((probabilities > 0).sum()),
        "effective_chunks": float(1 / np.square(probabilities).sum()),
        "domain_mass": {str(k): float(probabilities[train["domain"] == k].sum()) for k in np.unique(train["domain"])},
        "bucket_mass": {str(k): float(probabilities[train["bucket"] == k].sum()) for k in np.unique(train["bucket"])}})
    model, stages["final"] = train_stage(cfg, train, dev, directory, "final", seed, fingerprint, probabilities=probabilities)
    metrics = evaluate(model, eval_path, directory, cfg)
    result = {"status": "complete", "arm": arm, "seed": seed, "fingerprint": fingerprint,
              "stages": stages, "metrics": metrics, "completed_at": utc_now(),
              "invocation_wall_seconds": time.monotonic() - started,
              "total_stage_wall_seconds": sum(s["wall_seconds"] for s in stages.values()),
              "total_training_flops_proxy": sum(s["flops_proxy"] for s in stages.values()),
              "heldout_nll_sha256": sha256(directory / "heldout_nll.npz")}
    atomic_json(directory / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true", help="Use only the frozen max_recoveries allowance")
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text()))
    cfg.validate()
    torch.set_num_threads(cfg.num_threads)
    if torch.device(cfg.device).type == "cuda":
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the explicitly allocated GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("this harness requires exactly one visible allocated GPU")
        if cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 unsupported; freeze a separate fp32 configuration")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Kernel release on process exit prevents overlapping writers without stale PID files.
    with open(args.output_dir / ".writer.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = {name: sha256(args.data_dir / name) for name in ("train.npz", "dev.npz", "eval.npz")}
        data_manifest = args.data_dir / "manifest.json"
        if data_manifest.exists():
            frozen_files = json.loads(data_manifest.read_text()).get("files", {})
            if any(frozen_files.get(name, digest) != digest for name, digest in inputs.items()):
                raise ValueError("data files differ from the frozen preparation manifest")
        identity = {"config": asdict(cfg), "data_sha256": inputs, "train_source_sha256": sha256(Path(__file__))}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        frozen_path = args.output_dir / "frozen_config.json"
        if frozen_path.exists():
            if json.loads(frozen_path.read_text())["fingerprint"] != fingerprint:
                raise ValueError("configuration/data/source changed; use a new prospective output directory")
        else:
            atomic_json(frozen_path, {**identity, "fingerprint": fingerprint, "frozen_at": utc_now(),
                                     "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                                                 "python": platform.python_version(), "cuda": torch.version.cuda,
                                                 "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}})
        train, dev = load_training_data(args.data_dir, cfg)
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = {"fingerprint": fingerprint, "expected_cells": len(cfg.arms) * len(cfg.seeds),
                        "cells": [{"arm": arm, "seed": seed, "status": "pending", "recoveries": 0}
                                  for seed in cfg.seeds for arm in cfg.arms]}
            if os.environ.get("ZIPMIX_RUN_ID"):
                manifest["run_id"] = os.environ["ZIPMIX_RUN_ID"]
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("manifest identity mismatch")
        atomic_json(manifest_path, manifest)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        for cell in manifest["cells"]:
            directory = args.output_dir / f"{cell['arm']}_seed{cell['seed']}"
            if cell["status"] == "complete":
                result = json.loads((directory / "metrics.json").read_text())
                if result["fingerprint"] != fingerprint or result["heldout_nll_sha256"] != sha256(directory / "heldout_nll.npz"):
                    raise ValueError("complete cell artifact identity failed")
                continue
            if cell["status"] in ("failed", "interrupted", "running"):
                if (not args.retry_failed or cell["recoveries"] >= cfg.max_recoveries
                        or cell.get("failure_kind") == "numerical"):
                    continue
                cell.setdefault("recovery_history", []).append({
                    "prior_status": cell["status"], "prior_error": cell.get("error"),
                    "resumed_at": utc_now(), "recovery": cell["recoveries"] + 1})
                cell["recoveries"] += 1
            cell.update(status="running", started_at=utc_now())
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(manifest_path, manifest)
            try:
                run_cell(cell["arm"], cell["seed"], cfg, train, dev, args.data_dir / "eval.npz", directory, fingerprint)
                cell.update(status="complete", completed_at=utc_now(), metrics_file=str(directory / "metrics.json"))
            except (Exception, KeyboardInterrupt) as error:
                kind = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                failure = {"status": kind, "failure_kind": failure_kind(error), "at": utc_now(),
                           "error": str(error), "traceback": traceback.format_exc()}
                cell.update(status=kind, failure_kind=failure["failure_kind"], error=str(error), failed_at=utc_now())
                atomic_json(directory / f"failure_{cell['recoveries']}.json", failure)
                print(json.dumps({"cell": directory.name, **failure}), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if isinstance(error, KeyboardInterrupt):
                    atomic_json(manifest_path, manifest)
                    raise
            finally:
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
        counts = {status: sum(c["status"] == status for c in manifest["cells"])
                  for status in ("pending", "running", "complete", "failed", "interrupted")}
        manifest["counts"] = counts
        manifest["cells_with_recovery"] = sum(c["recoveries"] > 0 for c in manifest["cells"])
        manifest["full_matrix_complete"] = counts["complete"] == manifest["expected_cells"]
        manifest["full_clean_matrix"] = manifest["full_matrix_complete"] and manifest["cells_with_recovery"] == 0
        atomic_json(manifest_path, manifest)
        print(json.dumps({"expected": manifest["expected_cells"], **counts}), flush=True)
        return 0 if manifest["full_matrix_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
