# Zip-Mix agent instructions

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/AGENTS.md>

**TLDR:** Refresh and follow the shared agent rules. Zip-Mix is an experimental validation-guided training-data mixture project; distinguish hypotheses from measured generalization.

Run `git -C ~/agents-config pull --ff-only` and read `~/agents-config/INDEX_RULES.md` before starting a new task. Use the local copy if the remote is unavailable.

Read `README.md` and `experiments/README.md`. Experiment-specific code, configuration, data manifests, runbooks and results belong together under their numbered experiment. Raw corpora, model weights, checkpoints and private host details stay ignored or in explicitly linked private storage. Preserve unrelated staged and unstaged work.

Active methods must distinguish the original maximum-similarity Zip-Mix definition from published ZIP-FIT's mean aggregation. DoReMi requires a reference model and excess loss; raw-loss reweighting is not DoReMi. Equal final training tokens do not imply equal end-to-end compute. Never label a compact implementation as a paper-scale reproduction or claim superiority before independent evaluation.

Use separate training, development and evaluation records. Freeze full method-by-seed manifests before training. Keep missing and failed cells visible, preserve healthy remote work across coordinator boundaries, and report uncertainty across independent training seeds. Review with a model only when the user explicitly requests it; deterministic verification always applies.

The current experiment homes and latest outcomes are linked in `experiments/README.md`. Paper drafts live in `paper_latex_and_notes/`; load the current shared research-writing instructions before editing them. Do not edit paper claims merely because a small screening run finishes.
