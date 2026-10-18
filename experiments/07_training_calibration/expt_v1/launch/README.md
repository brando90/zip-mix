# Experiment 07 launchers

**Doc link:** <https://github.com/brando90/zip-mix/blob/main/experiments/07_training_calibration/expt_v1/launch/README.md>

**TLDR:** Three ways to run the same two stages on one A100 (graphics processing unit): a SNAP node, a rented cloud box, or Modal. Run Stage A first; Stage B starts only from Stage A's passed `decision.json`. Nothing has been launched.

| Where | Stage A | Stage B | Notes |
|---|---|---|---|
| SNAP node (skampere1, …) | `launch/run_stage.sh calibration` | `launch/run_stage.sh stage_b` | Run from your own SSH session (Kerberos + Duo). Picks the first idle GPU (or set `GPU=<index>`), works under `/lfs/<host>/0/$USER/zipmix_exp07`, survives logout. |
| Lambda / any 1× A100 box | `bash launch/lambda_bootstrap.sh calibration` | `bash launch/lambda_bootstrap.sh stage_b` | Clones this branch, reuses Lambda Stack's CUDA PyTorch, then calls `run_stage.sh`. Keep the same disk for both stages, then terminate the instance. |
| Modal | `modal run experiments/07_training_calibration/expt_v1/launch/modal_app.py --stage calibration` | `... --stage stage_b` | A100-40GB, 4-hour timeout, everything on the `zipmix-exp07` volume; fetch results with `modal volume get zipmix-exp07 runs/calibration/decision.json .` |

Each launcher prepares the data first if it is missing (CPU, about 2 minutes; it refuses to continue unless Experiment 05's identities and the design receipt reproduce), then trains, then writes `decision.json` (Stage A) or `analysis.json` (Stage B) next to the ledger. Expected time: Stage A ≈ 1.5 and Stage B ≈ 1.2 A100-hours.

Check progress: `tail -f <work>/runs/<stage>/train.log` or read `ledger.json`. A crashed or preempted run resumes by launching again (on SNAP or Lambda remove `runs/<stage>/launch.claim` first); each cell resumes at most once, as frozen in [PROTOCOL.md](../PROTOCOL.md).

**TLDR-end:** [zip-mix: exp07 launch] Same two stages on SNAP, a 1× A100 cloud box or Modal; Stage A first, Stage B only after a passed decision.
