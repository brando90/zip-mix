#!/usr/bin/env bash
# Submit Experiment 07 on FarmShare or Sherlock (Slurm). Prints each job id and appends it to
# $ZIPMIX_WORK/jobs_<step>.txt.
#
#   slurm_submit.sh prep          # CPU: venv, data hash check, model download, CPU tests (~10 min)
#   slurm_submit.sh calibration   # GPU: Stage A, 20 cells + base (one L40S, a few hours)
#   slurm_submit.sh all           # prep, then Stage A after it succeeds
#   slurm_submit.sh stage_b       # GPU: Stage B, 15 cells + base; refused unless Stage A's decision.json passed
#
# The data comes from the upload bundle unpacked as $ZIPMIX_WORK/data (see README.md); environment variables as in
# slurm_common.sh. Set FORCE=1 to submit while a job of the same step is still queued or running.
set -euo pipefail
step=${1:?usage: slurm_submit.sh prep|calibration|all|stage_b}
export ZIPMIX_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ZIPMIX_CODE/launch/slurm_common.sh"
mkdir -p "$ZIPMIX_WORK/logs"
log=(-o "$ZIPMIX_WORK/logs/%x-%j.out")

record() {  # step job_id
  local line="$(date +%Y-%m-%dT%H:%M:%S%z) $1 job $2 on $ZIPMIX_CLUSTER, code $(zm_commit)"
  echo "$line" >>"$ZIPMIX_WORK/jobs_$1.txt"
  echo "$line"
}

guard() {  # refuse a duplicate of a step that is still queued or running
  local active
  active=$(squeue -h -u "$USER" -n "zm07-$1" -o %i 2>/dev/null | tr '\n' ' ')
  [[ -z $active || ${FORCE:-0} == 1 ]] || zm_die "zm07-$1 already queued/running: $active (FORCE=1 to override)"
}

submit_prep() {
  guard prep
  if [[ $ZIPMIX_CLUSTER == sherlock ]]; then
    # Sherlock's dev partition refuses sbatch: run the same script in the foreground there (Slurm mail still comes).
    srun -p dev -c 4 --mem=16G -t 01:00:00 -J zm07-prep "${ZM_MAIL[@]}" bash "$ZIPMIX_CODE/launch/slurm_prep.sbatch" \
      >"$ZIPMIX_WORK/logs/zm07-prep-dev.out" 2>&1 || { tail -n 30 "$ZIPMIX_WORK/logs/zm07-prep-dev.out"; zm_die "prep failed"; }
    record prep "$(sacct -n -X -u "$USER" --name zm07-prep -o JobID -S now-2hours | tail -1 | tr -d ' ')"
    PREP_ID=
  else
    # wheat-01 lacks python3.12-venv.
    PREP_ID=$(sbatch --parsable -p normal --exclude=wheat-01 "${log[@]}" "$ZIPMIX_CODE/launch/slurm_prep.sbatch")
    record prep "$PREP_ID"
  fi
}

submit_stage() {  # stage [dependency job id]
  local stage=$1 after=${2:-} id limit
  guard "$stage"
  if [[ $stage == stage_b ]]; then
    zm_activate
    zm_gate_passed "$ZIPMIX_WORK/runs/calibration/decision.json" || zm_die "Stage A gate did not pass (or is not done); Stage B is not submitted"
    limit=08:00:00
  else
    limit=10:00:00
  fi
  id=$(sbatch --parsable "${ZM_GPU_FLAGS[@]}" -t "$limit" -J "zm07-$stage" "${log[@]}" \
       ${after:+--dependency=afterok:$after} "$ZIPMIX_CODE/launch/slurm_stage.sbatch" "$stage")
  record "$stage" "$id"
}

case $step in
  prep) submit_prep ;;
  calibration) submit_stage calibration ;;
  all) submit_prep; submit_stage calibration "$PREP_ID" ;;
  stage_b) submit_stage stage_b ;;
  *) zm_die "unknown step: $step" ;;
esac
echo "logs: $ZIPMIX_WORK/logs   check: squeue -u $USER -n zm07-prep,zm07-calibration,zm07-stage_b"
