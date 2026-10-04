# Experiment 05: validation-guided supervised fine-tuning screen

Status on 10-04-2026: implemented and data frozen; deterministic checks passed; training has not been launched by the implementation agent. This is a small mechanism screen, not a claim of optimal training or superiority over current methods.

The experiment asks whether compression-selected mixtures improve a pretrained model's multiple-choice science accuracy compared with equal-budget sampling controls. It adapts the pinned Qwen2.5-0.5B base checkpoint using 2,048 supervised answer-label tokens per training cell. Six methods and three sampling seeds produce 18 cells, plus one base-checkpoint evaluation. All model computation uses local pretrained weights; no paid model-service calls are used.

| Item | Frozen choice |
|---|---|
| Training pool | At most 3,000 eligible examples from each of SciQ, OpenBookQA, AI2 Reasoning Challenge (ARC-Challenge), and CommonsenseQA training splits |
| Alignment | 64 hash-ranked eligible SciQ validation questions with answer choices, without answer labels |
| Target evaluation | 500 hash-ranked eligible SciQ test examples |
| Retention evaluation | 500 hash-ranked eligible CommonsenseQA validation examples |
| Methods | Sample-proportional, source-uniform, Compel, historical maximum-score ZipMix, direct mean-score ZIP-FIT, shuffled-score ZipMix |
| Training | Full fine-tuning with float32 master parameters and bfloat16 forward autocasting, 128 steps × 16 examples, one graphics processing unit (GPU), seeds 0/1/2 |
| Primary outcome | Multiple-choice accuracy on SciQ; answer-label negative log likelihood is secondary |
| Decision rule | Complete and report the full screen; use direction, uncertainty, source exposure, and selection cost to design a separate confirmation experiment |

The CommonsenseQA training split is in the candidate pool, so its validation accuracy measures untargeted-task retention or transfer, not a task family excluded from training. The backbone may have prior exposure to either benchmark. The three-seed screen is descriptive: the smallest attainable exact two-sided paired sign-flip p-value is 0.25. Final superiority remains unresolved until larger, prospectively frozen comparisons include faithful relevant competitors.

All examples use a fixed multiple-choice prompt and a single answer-letter continuation. Prompt tokens are excluded from the loss. Every candidate answer is asserted to be exactly one tokenizer token, including the prompt boundary. Equal training-example counts therefore equal supervised-token counts. Input-token exposure and total computation vary with selected question length and are reported separately.

The Compel interval `[0.65, 0.80]` originated for longer pretraining documents. Applying it to short question-and-choice strings is an explicit transfer baseline with potentially sparse support. The pipeline refuses an empty band before freezing data and never silently relaxes it. Maximum similarity is the repository's historical ZipMix variant; the published ZIP-FIT algorithm uses the mean and that mean is used for direct selection. This screen does not isolate aggregation choice from bucket versus direct selection.

Preparation produced 10,116 training candidates, 64 alignment views, and 500 items per evaluation set. Compel retains 150 candidates (1.48%); repeated exposure is therefore expected. Historical ZipMix assigns 29.6563% mass to SciQ, compared with 29.6560% under sample-proportional sampling and 29.6599% under the source-and-length permutation control. These are exact descriptive properties of the frozen finite pool, not performance estimates (p-val=n/a); the weak target-mass change is a reason to test the hypothesis carefully, not to assume a win. See [data manifest](expt_v1/data_manifest.json).

Review measurement on this frozen pool: the LZ4 ratio and maximum alignment are dominated by string length (Spearman correlation with byte length -0.92 and -0.88; 87.4% of candidates have an LZ4 ratio above 1 because framing overhead exceeds savings). The five compression bins are therefore close to length quintiles, and ZipMix upweights shorter questions. The source-by-length permutation control differs from ZipMix by total variation 0.0025, whereas ZipMix differs from sample-proportional by 0.0431. A ZipMix-versus-shuffled difference in this version therefore cannot separate validation alignment from length. Length-controlled scoring or a length-matched mismatched-target control needs a separate prospective version. Details: [review report](../04_validation_guided_zipmix/qa/opus55_max_review.md).

```text
05_validation_guided_sft/
  README.md
  results.md
  CKPT_validation_guided_sft.md
  expt_v1/
    PROTOCOL.md
    cc.md
    common.py
    prepare_sft.py
    train_sft.py
    analyze_sft.py
    test_sft.py
    test_analysis.py
    data/              # ignored pinned data, tokenizer, manifest, selection weights
    runs/              # ignored model checkpoints, predictions, durable ledgers
```

Dependencies are NumPy, SciPy, PyTorch, Transformers, Datasets, Hugging Face Hub, LZ4, and pytest for deterministic tests. The verified Stanford Network Analysis Project (SNAP) runtime is described in [Experiment 04 preflight](../04_validation_guided_zipmix/preflight.md). Large caches and checkpoints belong on the execution machine's local scratch storage, with exact private paths recorded by the coordinator rather than published here.

```bash
python experiments/05_validation_guided_sft/expt_v1/prepare_sft.py --workers 4
python -m pytest experiments/05_validation_guided_sft/expt_v1/test_sft.py experiments/05_validation_guided_sft/expt_v1/test_analysis.py
CUDA_VISIBLE_DEVICES=0 python experiments/05_validation_guided_sft/expt_v1/train_sft.py --data experiments/05_validation_guided_sft/expt_v1/data --output experiments/05_validation_guided_sft/expt_v1/runs/screen
python experiments/05_validation_guided_sft/expt_v1/analyze_sft.py --run experiments/05_validation_guided_sft/expt_v1/runs/screen
```

Device `0` is an example; the coordinator must verify availability and bind exactly one suitable device. Dataset and model revisions are pinned in `common.py`. Preparation refuses to overwrite an existing output directory, and training refuses a changed manifest/code fingerprint for an existing run. Restarts resume the same logical cell within its fixed bounds, record replayed work, and preserve failed rows.

The user-requested Opus 5.5 maximum-effort review of Experiments 04 and 05 is recorded in [its report](../04_validation_guided_zipmix/qa/opus55_max_review.md).

Sources: [Qwen model](https://huggingface.co/Qwen/Qwen2.5-0.5B), [SciQ](https://huggingface.co/datasets/allenai/sciq), [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa), [ZIP-FIT algorithm](https://arxiv.org/html/2410.18194v2#S2), [Compel](https://github.com/stair-lab/compel).
