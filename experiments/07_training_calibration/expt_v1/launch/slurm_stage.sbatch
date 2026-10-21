#!/usr/bin/env bash
# GPU job: one Experiment 07 stage, every cell in order on ONE bf16 GPU, then its analysis.
#   calibration  Stage A: 20 cells + base -> runs/calibration/decision.json
#   stage_b      Stage B: 15 cells + base -> runs/stage_b/analysis.json; refused unless decision.json passed
# One job per stage, not an array: the trainer keeps every cell in one ledger under an exclusive lock and the
# analysis reads that ledger. Resubmitting resumes from the ledger (one recorded infrastructure resume per cell).
# Submit through slurm_submit.sh, which adds the cluster's GPU flags and the stage's time limit.
#SBATCH --job-name=zm07-stage
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
set -euo pipefail
stage=${1:?usage: slurm_stage.sbatch calibration|stage_b}
[[ $stage == calibration || $stage == stage_b ]] || { echo "unknown stage: $stage" >&2; exit 64; }
: "${ZIPMIX_CODE:?submit through launch/slurm_submit.sh}"
source "$ZIPMIX_CODE/launch/slurm_common.sh"
zm_activate
[[ -f $ZIPMIX_WORK/env_ok.json ]] || zm_die "no $ZIPMIX_WORK/env_ok.json; run: slurm_submit.sh prep"
echo "$stage ${SLURM_JOB_ID:-local} on $(hostname -s) ($ZIPMIX_CLUSTER), code $(zm_commit)"
nvidia-smi -L
"$PY" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "need a bf16 CUDA GPU"; print(torch.cuda.get_device_name(0))'
run=$ZIPMIX_WORK/runs/$stage
mkdir -p "$run"
extra=()
if [[ $stage == stage_b ]]; then
  decision=$ZIPMIX_WORK/runs/calibration/decision.json
  zm_gate_passed "$decision" || exit 66
  extra=(--decision "$decision")
fi
cd "$ZIPMIX_CODE"
set +e
"$PY" train_calibration.py --data "$ZIPMIX_WORK/data/$stage" --output "$run" --cache-dir "$HF_HOME/hub" "${extra[@]}"
status=$?
# The analysis also runs after a partial run; it then reports the missing cells instead of a result.
if [[ -f $run/ledger.json ]]; then "$PY" analyze_calibration.py --run "$run" --stage "$stage"; fi
set -e
echo "TRAIN_EXIT=$status"
exit $status
