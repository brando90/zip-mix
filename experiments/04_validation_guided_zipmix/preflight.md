# Zip-Mix validation-guided experiment preflight

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/04_validation_guided_zipmix/preflight.md>

**TLDR:** One isolated A100 runtime is verified for both experiments, including an actual disposable optimizer step. Private execution paths remain in the deployment receipt.

Checked 10-04-2026. The Stanford Network Analysis Project (SNAP) compute route is available; a dedicated source checkout and isolated single-device environment are ready for the final experiment inputs. No training result or superiority claim follows from this preflight.

## Compute and ownership

- A directly accessible machine has NVIDIA A100 graphics processing units (GPUs), each with 80 GiB device memory. Five devices were idle at the initial observation; availability must be checked immediately before launch.
- Start with one device, selected explicitly through `CUDA_VISIBLE_DEVICES`, and run all frozen experiment cells sequentially. Measure throughput, peak memory and utilization before changing resource allocation.
- The selected machine's local scratch filesystem has approximately 14 TiB free. The shared filesystem has approximately 636 GiB remaining but less than 1% free, so package installations, dataset caches and checkpoints belong on local scratch. Only the small owned source checkout and compact receipts should use shared storage.
- A dedicated source checkout and an isolated Python environment were created. No existing Zip-Mix process was found during the initial check. Other users' and projects' processes were left running.
- The documented direct Secure Shell (SSH) job route applies to the selected machine. The Slurm controller was not verified because host trust was unavailable; this does not prevent use of the already verified direct route.

## Environment

The isolated Python 3.12.3 environment contains 65 installed packages and occupies approximately 7.2 GB of local scratch. Verified core versions:

| Package | Version |
|---|---|
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.57.6 |
| Datasets | 4.8.4 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| Tokenizers | 0.22.2 |
| PyArrow | 24.0.0 |
| LZ4 | 4.4.5 |

With the Compute Unified Device Architecture (CUDA) 12.8 runtime and exactly one visible A100, both a matrix identity check and a tiny GPT-2 forward computation passed. Logit dimensions were `[1, 16, 128]`, all logits were finite, and peak allocated tensor memory was 9,698,816 bytes. This was an environment check, not a training or throughput measurement. No GPU process was intentionally retained.

Run the isolated interpreter with ambient `PYTHONPATH` removed and `PYTHONNOUSERSITE=1`. The complete installed package freeze and computation receipt are retained in the private runtime directory.

Existing local caches include model weights for GPT-2, OPT-125M, Pythia-1.4B and Qwen3-1.7B. Cache presence alone does not verify compatibility or a reproducible model revision. Freeze the chosen model and tokenizer revision separately before any post-training experiment.

## Agent instructions and requested review

The remote agent-rules checkout was refreshed to revision `2b5334965c23e6a0396d2c47078d68b1b127bc22`. The canonical global Codex and Claude instruction files are nonempty and resolve to the intended shared rules. Existing running agents were not restarted or claimed to have reloaded new content. The preparation agent read the refreshed relevant rules.

Actual remote Codex settings specify full filesystem access and no routine approvals. The Claude full-access wrapper prepends the supported permission-bypass option. Any new agent dispatch must still pin the intended model and effort explicitly.

The local Claude Code command-line interface (CLI), version 2.1.289, reports a logged-in subscription session and supports explicit model selection and `--effort max`. Its installed binary contains the requested exact identifier `claude-opus-5-5`. A useful quality assurance (QA) review must verify that model's actual entitlement and record the resolved model; binary/catalog evidence alone is insufficient. No review or probe model call was made during preflight. Remaining subscription allowance is unknown.

## Before starting the frozen run

1. Transfer the final source, manifest and data, and verify their hashes remotely.
2. Recheck the chosen GPU, bind a single device, and record the process identity.
3. Launch with a durable task-owned process and a tested watchdog; cover every declared method-by-seed cell and preserve failures in the denominator.
4. Record an initial throughput estimate and resource measurements, then allow the full bounded run to complete.

Private machine paths, routes and account receipts are stored outside this public repository. This preflight does not establish the watchdog, submit training, or measure a generalization gain.

## Completed runtime checks

The 20-step pretraining timing completed 1/1 engineering cell in 4.99 seconds (3.29 training seconds); no timing result enters scientific inference. The pinned Qwen2.5-0.5B model also completed one disposable synthetic optimizer step with float32 master weights and bfloat16 autocast: 0.639 seconds, 9,975,041,536 peak allocated bytes, all 1,024 sampled weight elements changed. These are engineering measurements, not benchmark results.

Remote main command-line defaults now read `gpt-6-astra` / `ultra`, `danger-full-access` / `never`; the private backup preserves prior configuration and every unrelated parsed field was checked unchanged. Existing workers were not restarted; future launches read the updated settings. The requested review actually ran as `claude-opus-5-5` with the explicit `--effort max` flag; see the [review report](qa/opus55_max_review.md).
