#!/usr/bin/env bash
# Launch one Experiment 07 stage on ONE free GPU of the host you are logged in to
# (a SNAP node such as skampere1, or a rented 1x A100 box). Survives logout.
#
#   launch/run_stage.sh calibration          # Stage A: 20 cells + base, ~1.5 A100-hours
#   launch/run_stage.sh stage_b              # Stage B: 15 cells + base, ~1.2 A100-hours,
#                                            # only after Stage A wrote a passed decision.json
#
# Environment (all optional):
#   ZIPMIX_WORK   private work root; default /lfs/$(hostname -s)/0/$USER/zipmix_exp07
#                 when that /lfs disk exists, else $HOME/zipmix_exp07
#   PYTHON_BIN    interpreter with requirements.txt installed; default: $ZIPMIX_WORK/venv,
#                 created on first use with --system-site-packages
#   GPU           device index to bind; default: the first GPU with <1 GiB used
#   HF_HOME       model/dataset cache; default $ZIPMIX_WORK/hf
# Data, runs, weights and logs stay under ZIPMIX_WORK, never in the repository.
set -euo pipefail
stage=${1:?usage: run_stage.sh calibration|stage_b}
[[ $stage == calibration || $stage == stage_b ]] || { echo "unknown stage: $stage" >&2; exit 64; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
host=$(hostname -s)
if [[ -z ${ZIPMIX_WORK:-} ]]; then
  if [[ -d /lfs/$host/0/$USER ]]; then ZIPMIX_WORK=/lfs/$host/0/$USER/zipmix_exp07; else ZIPMIX_WORK=$HOME/zipmix_exp07; fi
fi
umask 077
mkdir -p "$ZIPMIX_WORK"
export HF_HOME=${HF_HOME:-$ZIPMIX_WORK/hf}
if [[ -z ${PYTHON_BIN:-} ]]; then
  if [[ ! -x $ZIPMIX_WORK/venv/bin/python ]]; then
    python3 -m venv --system-site-packages "$ZIPMIX_WORK/venv"
    "$ZIPMIX_WORK/venv/bin/pip" install -q -r "$here/requirements.txt"
  fi
  PYTHON_BIN=$ZIPMIX_WORK/venv/bin/python
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "need a bf16 CUDA GPU"'
if [[ ! -f $ZIPMIX_WORK/data/stage_b/manifest.json ]]; then
  # CPU only, ~2 minutes; verifies Experiment 05 identities and the design receipt.
  (cd "$here" && "$PYTHON_BIN" prepare_calibration.py --output "$ZIPMIX_WORK/data")
fi
if [[ -z ${GPU:-} ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1024 {print $1; exit}')
  [[ -n $GPU ]] || { echo "no idle GPU on $host; set GPU=<index> explicitly" >&2; exit 75; }
fi
run=$ZIPMIX_WORK/runs/$stage
mkdir -p "$run"
if ! mkdir "$run/launch.claim" 2>/dev/null; then
  echo "already launched: $run (resume by removing $run/launch.claim; the ledger resumes each cell once)" >&2
  exit 73
fi
extra=()
if [[ $stage == stage_b ]]; then
  decision=$ZIPMIX_WORK/runs/calibration/decision.json
  [[ -f $decision ]] || { rmdir "$run/launch.claim"; echo "missing $decision; finish Stage A first" >&2; exit 66; }
  extra=(--decision "$decision")
fi
cd "$here"
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c '
  "$0" train_calibration.py --data "$1/data/$2" --output "$3" --cache-dir "$HF_HOME" "${@:4}"
  "$0" analyze_calibration.py --run "$3" --stage "$2"
' "$PYTHON_BIN" "$ZIPMIX_WORK" "$stage" "$run" "${extra[@]}" </dev/null >"$run/train.log" 2>&1 &
echo $! >"$run/launch.pid"
printf 'Launched %s on %s GPU %s: pid %s\nlog: %s/train.log\nresult: %s/%s\n' "$stage" "$host" "$GPU" "$!" "$run" "$run" \
  "$([[ $stage == calibration ]] && echo decision.json || echo analysis.json)"
