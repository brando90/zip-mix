# Experiment 07 launchers

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_training_calibration/expt_v1/launch/README.md>

**TLDR:** Four ways to run the same two stages on one GPU (graphics processing unit): a SNAP node, a rented A100 box, Modal, or Slurm on FarmShare/Sherlock (one bf16 GPU). Run Stage A first; Stage B starts only from Stage A's passed `decision.json`. Nothing has been launched.

| Where | Stage A | Stage B | Notes |
|---|---|---|---|
| SNAP node (skampere1, …) | `launch/run_stage.sh calibration` | `launch/run_stage.sh stage_b` | Run from your own SSH session (Kerberos + Duo). Picks the first idle GPU (or set `GPU=<index>`), works under `/lfs/<host>/0/$USER/zipmix_exp07`, survives logout. |
| Lambda / any 1× A100 box | `bash launch/lambda_bootstrap.sh calibration` | `bash launch/lambda_bootstrap.sh stage_b` | Clones this branch, reuses Lambda Stack's CUDA PyTorch, then calls `run_stage.sh`. Keep the same disk for both stages, then terminate the instance. |
| FarmShare or Sherlock (Slurm) | `launch/slurm_submit.sh all` (prep, then Stage A) | `launch/slurm_submit.sh stage_b` | One sbatch job per stage on one bf16 GPU (FarmShare `-p gpu --qos=gpu`, every GPU an L40S; Sherlock `-p gpu -C 'GPU_SKU:L40S\|GPU_SKU:H100_SXM5'`). Data comes from the upload bundle, hash-checked. Job ids go to `<work>/jobs_*.txt`; `launch/slurm_collect.sh <stage>` packs results for download. See below. |
| Modal | `modal run experiments/07_training_calibration/expt_v1/launch/modal_app.py --stage calibration` | `... --stage stage_b` | A100-40GB, 4-hour timeout, everything on the `zipmix-exp07` volume; fetch results with `modal volume get zipmix-exp07 runs/calibration/decision.json .` |

Each launcher prepares the data first if it is missing (CPU, about 2 minutes; it refuses to continue unless Experiment 05's identities and the design receipt reproduce), then trains, then writes `decision.json` (Stage A) or `analysis.json` (Stage B) next to the ledger. Expected time: Stage A ≈ 1.5 and Stage B ≈ 1.2 A100-hours.

Check progress: `tail -f <work>/runs/<stage>/train.log` or read `ledger.json`. A crashed or preempted run resumes by launching again (on SNAP or Lambda remove `runs/<stage>/launch.claim` first); each cell resumes at most once, as frozen in [PROTOCOL.md](../PROTOCOL.md).

## FarmShare or Sherlock (Slurm)

Neither cluster can run `prepare_calibration.py` into the frozen identities: it stamps the preparation time and runtime into the manifests and rewrites the committed `calibration_manifest.json` and `stage_b_manifest.json`. So the prepared data travels in an upload bundle built where it was prepared:

```bash
launch/make_upload_bundle.sh <prepared data dir> zm07_bundle.tgz   # repo at HEAD + data, as zipmix_exp07/{repo,data}
```

On the cluster (FarmShare: `<work>` = `~/zipmix_exp07`, venv `~/venvs/zipmix`, `HF_HOME=~/hf_cache`; Sherlock: `$SCRATCH/zipmix_exp07`, its own venv inside it, `HF_HOME=$SCRATCH/hf_cache`):

```bash
tar xzf zm07_bundle.tgz -C ~          # Sherlock: -C $SCRATCH
bash ~/zipmix_exp07/repo/experiments/07_training_calibration/expt_v1/launch/slurm_submit.sh all
```

- `prep` (CPU, about 10 minutes): builds the experiment's own venv (FarmShare: the torch build of `~/venvs/fs_hf`; Sherlock: `python/3.12.1`, wheel-only `torch<=2.6.0`), checks that the data manifests are byte-identical to the committed ones and every file hash matches, downloads the pinned model, runs the CPU tests and writes `<work>/env_ok.json`. On Sherlock it runs in the foreground on `dev`, which refuses sbatch.
- `calibration`: one GPU job (10-hour limit) runs base plus all 20 cells in order, then writes `runs/calibration/decision.json`. Not a job array: the trainer keeps one ledger under an exclusive lock and the analysis reads that ledger.
- `stage_b`: submitted only when `decision.json` verifies and passed; the job and the trainer check again. 8-hour limit.
- Every job mails END,FAIL to the submitting account's Slurm address (set `ZM_MAIL_USER=a@x,b@y` in the shell to send elsewhere; addresses are not stored in the repo). A failed or timed-out stage resumes by submitting it again (each cell resumes at most once).
- `launch/slurm_collect.sh calibration|stage_b` writes `<work>/exp07_<stage>_results.tgz` (no checkpoints or weights) and prints its OnDemand `?download=1` link.

**TLDR-end:** [zip-mix: exp07 launch] Same two stages on SNAP, a 1× A100 cloud box, Modal or Slurm (FarmShare/Sherlock); Stage A first, Stage B only after a passed decision.
