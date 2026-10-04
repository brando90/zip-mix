# Experiment 05: validation-guided supervised fine-tuning

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/05_validation_guided_sft/README.md>

Version 2 is the active prospective experiment. Updated 10-04-2026: the full 21-cell run started at 12:08 PDT on a dedicated A100 after 38 remote tests and all input/source hash checks passed. The unchanged-base evaluation completed; final paired outcomes remain pending.

The user-requested single Opus 5.5 maximum-effort quality-assurance review found that version 1's short-string compression scores mostly tracked length. [Version 1 remains an untrained calibration](expt_v1/STATUS.md). Version 2 repairs the scoring unit before training: every candidate and development view contains exactly 4,096 real bytes, with a matched wrong-target control. No artificial padding or repetition fills views.

The scientific question is whether changing the development-target compression prior improves held-out science accuracy at a fixed supervision budget. The current evidence establishes that the prior changes; it does not establish that training with it improves generalization.

| Item | Frozen version 2 choice |
|---|---|
| Backbone | Pinned Qwen2.5-0.5B base checkpoint |
| Training pool | 9,797 examples from SciQ, OpenBookQA, ARC-Challenge, and CommonsenseQA, arranged into 351 disjoint source-specific packs |
| True development target | Eight 4,096-byte SciQ validation packs |
| Wrong development target | Eight matched-byte CommonsenseQA packs from 256 reserved training records excluded from every arm |
| Target evaluation | 500 fixed SciQ test examples |
| Retention evaluation | 500 fixed CommonsenseQA validation examples |
| Mixtures | Sample-proportional, source-uniform, Compel, historical maximum-score ZipMix, mean-score direct ZIP-FIT adaptation, shuffled-score ZipMix, wrong-target ZipMix |
| Training | Seven methods × three seeds = 21 cells; 128 steps × 16 examples = 2,048 supervised answer-label tokens per cell |
| Precision | Full fine-tuning with float32 master parameters/optimizer states and bfloat16 forward autocasting |
| Outcome | Multiple-choice accuracy; answer-label negative log likelihood is secondary |

![Development-target mixture diagnostics](expt_v2/results/mixture_diagnostics.png)

**Changing the development target changes the frozen training mixture.** All scoring views contain 4,096 real bytes; values are exact finite-pool probabilities, not performance estimates. Source and pack-density confounds remain.

The true-target ZipMix mixture places 35.87% mass on SciQ, versus 26.93% for the wrong target and 30.53% for sample-proportional sampling. True-versus-wrong mixture total variation is 0.13953; true-versus-shuffled total variation is 0.06509. These describe the complete frozen pool exactly, not sampled performance estimates (p-val=n/a). All seven methods have support; Compel retains 9,722 of 9,797 examples. Its near-natural mixture is reported rather than made artificially selective.

The full [protocol](expt_v2/PROTOCOL.md), [public input manifest](expt_v2/data_manifest.json), [execution prompt](expt_v2/cc.md), [live results](results.md), and [checkpoint](CKPT_validation_guided_sft.md) are canonical. Raw public datasets, token arrays, model weights, and run directories remain ignored. Preserve them on execution-host scratch storage with private paths recorded by the coordinator.

```text
05_validation_guided_sft/
  README.md
  results.md
  CKPT_validation_guided_sft.md
  expt_v1/                 # preserved untrained calibration and original protocol
  expt_v2/
    PROTOCOL.md
    cc.md
    data_manifest.json
    common.py
    prepare_sft.py
    train_sft.py
    analyze_sft.py
    test_sft.py
    test_analysis.py
    test_packs.py
    results/mixture_diagnostics.png
    data/                  # ignored prepared input artifacts
    runs/                  # ignored weights, predictions, durable ledgers
```

Use the verified Stanford Network Analysis Project (SNAP) runtime described in [Experiment 04 preflight](../04_validation_guided_zipmix/preflight.md). Run each version's tests in a separate process because the versioned scripts intentionally have the same module names.

```bash
python experiments/05_validation_guided_sft/expt_v2/prepare_sft.py
python -m pytest experiments/05_validation_guided_sft/expt_v2/ -q
CUDA_VISIBLE_DEVICES=0 python experiments/05_validation_guided_sft/expt_v2/train_sft.py --data experiments/05_validation_guided_sft/expt_v2/data --output experiments/05_validation_guided_sft/expt_v2/runs/screen
python experiments/05_validation_guided_sft/expt_v2/analyze_sft.py --run experiments/05_validation_guided_sft/expt_v2/runs/screen
```

The device index is an example; the coordinator must recheck availability and bind exactly one suitable graphics processing unit. Preparation refuses to overwrite frozen inputs. The trainer checks input/code identities, saves recoverable state every 16 steps, and preserves every failed row. A durable external supervisor must cover the full manifest and require successful base evaluation as well as all training cells.

Three seeds permit descriptive paired-seed intervals, with weak small-sample precision; an exact two-sided sign test cannot attain p<0.25 at three non-tied pairs. No confirmatory significance or optimality claim follows from this screen. CommonsenseQA training data remain in the pool, so its evaluation is untargeted retention/transfer rather than an unseen family. The pretrained backbone may have benchmark exposure. Direct ZIP-FIT uses inherited pack scores here, and Compel is applied to packs of benchmark questions; neither is a faithful reproduction of its original setting. Comparisons against faithful DoReMi, DoGE, LESS, and current mixing competitors remain separate research stages.

Sources: [Qwen model](https://huggingface.co/Qwen/Qwen2.5-0.5B), [SciQ](https://huggingface.co/datasets/allenai/sciq), [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa), [ZIP-FIT algorithm](https://arxiv.org/html/2410.18194v2#S2), [Compel](https://github.com/stair-lab/compel).
