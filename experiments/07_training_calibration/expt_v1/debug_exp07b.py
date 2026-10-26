#!/usr/bin/env python3
"""Exp07b post-hoc DEBUG sweep (exploratory, not pre-registered): why was Exp07 Stage B null?

Reuses Exp07's trainer pieces (same model, 128 steps x 16 one-token labels, same eval sets) but samples
the 128x16 draws from arbitrary weights over the Stage B pool, so it can test:
  uniform      uniform over the whole pool (Exp07's recipe, no strata)
  global_tilt  UNSTRATIFIED Zip-Mix: w = exp(log4 * z), z = global z-score of (maximum - wrong_maximum)
  oracle_sciq  SciQ items only (upper bound: the target domain itself)
  anti_sciq    no SciQ items (lower bound / negative control)
Writes one JSON per cell + predictions under --output; resumable by skipping finished cells.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from train_calibration import (load_arrays, left_padded_batch, last_logits, learning_rate_scale, evaluate, new_model)
from calib_common import TRAIN


def make_weights(sel, pack, dom):
    pi = sel["pack_index"]
    contrast = (pack["maximum"] - pack["wrong_maximum"])[pi]
    z = (contrast - contrast.mean()) / contrast.std()
    w = {"uniform": np.ones(len(dom)), "global_tilt": np.exp(np.log(4) * z),
         "oracle_sciq": (dom == "sciq").astype(float) + 1e-12, "anti_sciq": (dom != "sciq").astype(float) + 1e-12}
    return {k: v / v.sum() for k, v in w.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)  # data/stage_b
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache-dir"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-6, 2e-5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+", default=["uniform", "global_tilt", "oracle_sciq", "anti_sciq"])
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda": assert torch.cuda.is_bf16_supported()
    man = json.loads((args.data / "manifest.json").read_text())
    train = load_arrays(args.data / "train.npz")
    ev = {n: load_arrays(args.data / f"{n}.npz", 32 if args.smoke else None)
          for n in ["holdout_target", "holdout_retention", "calibration_target", "calibration_retention"]}
    sel = load_arrays(args.data / "selection.npz"); pack = load_arrays(args.data / "pack_scores.npz")
    W = make_weights(sel, pack, train["domain"])
    steps, batch = (4, 4) if args.smoke else (TRAIN["steps"], TRAIN["batch_size"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "weights_summary.json").write_text(json.dumps(
        {a: {d: float(W[a][train["domain"] == d].sum()) for d in np.unique(train["domain"])} for a in W}, indent=1))
    ns = argparse.Namespace(smoke=args.smoke, cache_dir=args.cache_dir)
    if not (args.output / "base" / "result.json").exists():
        (args.output / "base").mkdir(exist_ok=True)
        m = new_model(ns, dev, 0)
        r = evaluate(m, ev, man["pad_token_id"], man["label_tokens"], 16, dev, args.output / "base")
        (args.output / "base" / "result.json").write_text(json.dumps(r, indent=1)); del m
    for lr in args.lrs:
        for arm in args.arms:
            for seed in args.seeds:
                cell = args.output / f"{arm}__lr{lr:g}__seed{seed}"
                if (cell / "result.json").exists(): continue
                cell.mkdir(exist_ok=True); t0 = time.time()
                model = new_model(ns, dev, seed); model.train()
                opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=tuple(TRAIN["betas"]), eps=TRAIN["eps"],
                                        weight_decay=TRAIN["weight_decay"])
                rng = np.random.default_rng(seed); drawn = []; losses = []
                for s in range(steps):
                    idx = rng.choice(len(train["ids"]), batch, replace=True, p=W[arm]); drawn.extend(idx.tolist())
                    inp, _ = left_padded_batch(train, idx, man["pad_token_id"], dev)
                    gold = torch.as_tensor(train["answer_token"][idx], dtype=torch.long, device=dev)
                    for g in opt.param_groups: g["lr"] = lr * learning_rate_scale(s, steps, TRAIN["warmup_steps"] if not args.smoke else 1)
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(last_logits(model, inp), gold); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["gradient_clip"]); opt.step(); losses.append(float(loss))
                r = evaluate(model, ev, man["pad_token_id"], man["label_tokens"], 16, dev, cell)
                (cell / "result.json").write_text(json.dumps({"arm": arm, "lr": lr, "seed": seed, "metrics": r,
                    "mean_train_loss": float(np.mean(losses)), "sciq_share_drawn": float(np.mean(train["domain"][drawn] == "sciq")),
                    "unique_drawn": len(set(drawn)), "seconds": time.time() - t0}, indent=1))
                print(arm, lr, seed, {k: round(v["accuracy"], 3) for k, v in r.items()}, flush=True)
                del model, opt


if __name__ == "__main__":
    main()
