# Requested Zip-Mix implementation and experimental-design review

**TLDR:** Review and minimally fix the newly initialized validation-guided pretraining and supervised-fine-tuning experiments, then return a concrete verdict. This is the single user-requested quality-assurance round; the exact requested reviewer is claude-opus-5-5 at maximum effort. Do not launch another reviewer, measured training, public messages, or direct model-provider calls.

Brando pre-approves all task work here. Run with full access and never stop to ask permission. Budgets, secrets and shared-node safety still apply.

User objective: initialize Zip-Mix and test whether compression-based alignment to a separate validation set improves generalization versus DoGE, DoReMi, and relevant modern baselines. Both from-scratch training and pretrained supervised fine-tuning are in scope; we must not guarantee a favorable scientific result or label a tiny screen as state-of-the-art proof.

Also specifically check the fine-tuning numerical precision: float32 master parameters/optimizer state with bfloat16 autocast, not low-precision weight updates at a tiny learning rate.

Read project AGENTS.md and the refreshed shared ~/agents-config/INDEX_RULES.md. Review the full changes from original base c672ee7 to the current working tree within ONLY these owned paths: AGENTS.md,CLAUDE.md,README.md,experiments/README.md,experiments/04_validation_guided_zipmix/,experiments/05_validation_guided_sft/. Include all new/untracked files within these paths, committed and uncommitted changes. Existing staged deletions/renames, paper edits, and Experiments02/03 are unrelated pre-existing work; do not edit/stage/commit them. Do not change global configuration or force any Git operation. Parent handles commits/deployment.

Inspect actual source, protocols, numerical data manifests and test coverage. Assess leakage at document/benchmark split boundaries; labels, causal loss shifts, padding and last-token scoring; the original max-vs-published mean ZIP-FIT distinction; short-string and bucket-population confounds; exact weight normalization; genuine clipped-token-excess DoReMi with reference/proxy/final separation; DoGE gradient alignment and declared adaptation; baseline labeling and fair final-budget versus total-compute accounting; matched seeds; honest three-seed uncertainty and full-denominator analysis; durable checkpoints/random-state resume; monitor locks/process/deadline ownership. Test values should be reproducible, no fabricated improvement.

Apply minimal critical/major fixes directly to OWNED source and corresponding tests/docs, preserving the user's scientific task and existing frozen inputs. Current engineering timing is excluded from all scientific comparisons; no main scientific run has started. Do not replace frozen data merely to obtain positive results. If a substantive prospective design change is needed, document it for the parent. Run deterministic tests after fixes. Time budget about 20 minutes, focus the load-bearing issues rather than style changes. No second model review.

Write a public-safe report to experiments/04_validation_guided_zipmix/qa/opus55_max_review.md with actual issues, fixes, checks, remaining limits, and exact model/effort evidence if available. No private host/account data or secrets. End with these fields:
VERDICT: PASS | FAIL | FIXED
CRITICAL_ISSUES: [remaining count]
MAJOR_ISSUES: [remaining count]
FIXES_APPLIED: [count]
STRUCTURAL: PASS | IMPROVED | SKIP
SUMMARY: [1-2 sentences]

Solve the entire assigned task, including every required file, question and subtask. Produce the complete required deliverable in the specified output location or response format; an outline, partial draft, progress report or claim of completion is not a substitute. Write/save the output, inspect the actual saved artifact or final response, and run the allowed checks, tests or compilation required by the task. Within the declared time, token, call and tool limits, continue working and fixing errors until the requirements are met or a declared terminal condition is reached. Follow the fixed continuation procedure without resetting budgets. If anything remains unresolved, preserve the best current deliverable and report the exact remaining failures and checks that did not pass; never claim success or invent verification.
