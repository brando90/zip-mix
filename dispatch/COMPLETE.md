# COMPLETE — Exp 07 null-result debug (attempt 3, Claude claude-fable-5-1, effort max)

Written: 10-21-2026 08:50 PDT (UTC 2026-10-21T15:50:56Z), after deliverables and verification.

- Branch: `codex-zipmix-e07-debug-10-21-2026`; deliverables commit `7c8885fadd63bd88e49300573eb97ebacbbe0834` pushed; remote head verified equal; `main` on GitHub unchanged at `449a98f`. No PR, no merge (not authorized).
- Report: `experiments/07_calibrated_conditional_sft_zipmix/debug_10-21-2026/DEBUG_REPORT.md` (copy: `dispatch/DEBUG_REPORT.md`). Live record: `dispatch/results.md`. Resume record: `dispatch/CKPT_e07_debug.md`. Acknowledgement: `dispatch/ACK.md`.
- Tests: `expt_v1/test_exp07.py` 13 passed (CPU; `env -u PYTHONPATH PYTHONNOUSERSITE=1 python -m pytest -q -p no:langsmith`); `py_compile` clean; frozen data 18/18 hashes unchanged after the session.
- Bugs fixed (local never-run trainer only): missing-file hash crash; `method_status` KeyError; selection metadata validated as weights. No bug found in the original run's code.
- Diagnostic: 15/15 cells + base at lr 5e-6, exit 0, 471 s wall on skampere2 GPU 1 (377 s summed training), GPU released; receipts under `debug_10-21-2026/diagnostic_lr5e-6/` with SHA256SUMS. Outcome: aligned − uniform +1.04 pp [−0.94, +3.02], aligned − shuffled +1.28 pp [−1.67, +4.23] on cal_target (N=500, 5 paired seeds, sign p=0.375 each); advance bar not met; inside the pre-registered +1.0 to +1.6 pp expectation.
- Original run preserved: branch `calib/exp07-fresh-calibration` @ `447e8a5` untouched; hashes of its ledgers/decision/analysis listed in the report; read-only extraction at `../orig_branch_ro/` (outside the repo, regenerable with `git archive`).
- Email: none sent by this worker (coordinator's watcher owns the terminal email).
- Cleanup owner: coordinator (parent chat). Safe to remove `/lfs/skampere2/0/brando9/snap_jobs/codex-zipmix-e07-debug-10-21-2026/` once the GitHub branch is confirmed; `orig_branch_ro/` and `expt_v1/runs_diag/` are regenerable; the 18 MB frozen `expt_v1/data/` is gitignored and originates from the transfer archive (sha256 in `dispatch/input.sha256`).
