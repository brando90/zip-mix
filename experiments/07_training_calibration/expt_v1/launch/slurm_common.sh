# Sourced by the Slurm launchers (slurm_*.sbatch, slurm_submit.sh, slurm_collect.sh). Not run directly.
#
# Environment (all optional):
#   ZIPMIX_CLUSTER  farmshare | sherlock; default: sherlock when $SCRATCH and /share/software exist, else farmshare
#   ZIPMIX_WORK     private work root holding data/, runs/, logs/ and jobs_*.txt;
#                   default $SCRATCH/zipmix_exp07 (Sherlock) or $HOME/zipmix_exp07 (FarmShare)
#   ZIPMIX_VENV     this experiment's own venv, never shared with another experiment;
#                   default $SCRATCH/zipmix_exp07/venv (Sherlock) or $HOME/venvs/zipmix (FarmShare)
#   HF_HOME         model cache; default $SCRATCH/hf_cache (Sherlock) or $HOME/hf_cache (FarmShare)
#   ZIPMIX_CODE     this expt_v1 directory; set by slurm_submit.sh because Slurm runs a spooled copy of each script
umask 077
ZIPMIX_CODE=${ZIPMIX_CODE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ -z ${ZIPMIX_CLUSTER:-} ]]; then
  if [[ -n ${SCRATCH:-} && -d /share/software ]]; then ZIPMIX_CLUSTER=sherlock; else ZIPMIX_CLUSTER=farmshare; fi
fi
case $ZIPMIX_CLUSTER in
  sherlock)
    ZIPMIX_WORK=${ZIPMIX_WORK:-$SCRATCH/zipmix_exp07}
    ZIPMIX_VENV=${ZIPMIX_VENV:-$SCRATCH/zipmix_exp07/venv}
    export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
    # bf16 GPUs only (the trainer refuses V100/P100 anyway).
    ZM_GPU_FLAGS=(-p gpu --gres=gpu:1 -C 'GPU_SKU:L40S|GPU_SKU:H100_SXM5')
    ZM_ONDEMAND=https://ondemand.sherlock.stanford.edu ;;
  farmshare)
    ZIPMIX_WORK=${ZIPMIX_WORK:-$HOME/zipmix_exp07}
    ZIPMIX_VENV=${ZIPMIX_VENV:-$HOME/venvs/zipmix}
    export HF_HOME=${HF_HOME:-$HOME/hf_cache}
    # Every FarmShare GPU is an L40S (bf16). QOS gpu: at most 4 GPUs at once, 2-day wall.
    ZM_GPU_FLAGS=(-p gpu --qos=gpu --gres=gpu:1)
    ZM_ONDEMAND=https://ondemand.farmshare.stanford.edu ;;
  *) echo "ZIPMIX_CLUSTER must be farmshare or sherlock, got: $ZIPMIX_CLUSTER" >&2; exit 64 ;;
esac
export ZIPMIX_CLUSTER ZIPMIX_WORK ZIPMIX_VENV ZIPMIX_CODE
PY=$ZIPMIX_VENV/bin/python
# Slurm mails the submitting account by default; set ZM_MAIL_USER=a@x,b@y to send elsewhere (kept out of the repo).
ZM_MAIL=(--mail-type=END,FAIL)
[[ -n ${ZM_MAIL_USER:-} ]] && ZM_MAIL+=(--mail-user="$ZM_MAIL_USER")

zm_die() { echo "ERROR: $*" >&2; exit 1; }

zm_activate() {
  # Sherlock's python/3.12.1 module provides the shared libraries the venv's interpreter links against.
  if [[ $ZIPMIX_CLUSTER == sherlock ]]; then module load python/3.12.1; fi
  [[ -x $PY && -f $ZIPMIX_VENV/.ready ]] || zm_die "no ready venv at $ZIPMIX_VENV; run: slurm_submit.sh prep"
}

zm_build_venv() {
  [[ -f $ZIPMIX_VENV/.ready ]] && { echo "venv ready: $ZIPMIX_VENV"; return; }
  mkdir -p "$(dirname "$ZIPMIX_VENV")"
  if [[ $ZIPMIX_CLUSTER == sherlock ]]; then
    # glibc 2.17: only manylinux2014 wheels install, so torch<=2.6.0 and no source builds.
    module load python/3.12.1
    python3 -m venv --clear "$ZIPMIX_VENV"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q --only-binary=:all: "torch<=2.6.0" "numpy==2.2.6" -r "$ZIPMIX_CODE/requirements.txt"
  else
    # Same torch build as the shared ~/venvs/fs_hf, which is known to run on the L40S nodes.
    local lock=$HOME/venvs/fs_hf/requirements.lock.txt torch_version=2.13.0
    [[ -f $lock ]] && torch_version=$(sed -n 's/^torch==\([0-9.]*\).*/\1/p' "$lock" | head -1)
    /usr/bin/python3.12 -m venv --clear "$ZIPMIX_VENV"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q "torch==${torch_version:-2.13.0}" -r "$ZIPMIX_CODE/requirements.txt"
  fi
  touch "$ZIPMIX_VENV/.ready"
}

zm_check_data() {
  # The shipped data must be the frozen data: byte-identical manifests and every file hash as recorded.
  local stage
  for stage in calibration stage_b; do
    [[ -f $ZIPMIX_WORK/data/$stage/manifest.json ]] || zm_die "missing $ZIPMIX_WORK/data/$stage; unpack the upload bundle first"
    cmp -s "$ZIPMIX_WORK/data/$stage/manifest.json" "$ZIPMIX_CODE/${stage}_manifest.json" \
      || zm_die "$ZIPMIX_WORK/data/$stage/manifest.json differs from the committed ${stage}_manifest.json"
  done
  (cd "$ZIPMIX_CODE" && "$PY" -c '
import sys
from pathlib import Path
import train_calibration as t
for stage in ("calibration", "stage_b"):
    print(stage, "manifest", t.verified_manifest(Path(sys.argv[1]) / stage)["manifest_id"], "all file hashes OK")
' "$ZIPMIX_WORK/data")
}

zm_gate_passed() {
  # Stage B is admitted only by a hash-verified, passed, non-smoke Stage A decision (the trainer checks again).
  [[ -f $1 ]] || { echo "missing $1; finish Stage A first" >&2; return 1; }
  (cd "$ZIPMIX_CODE" && "$PY" -c 'import sys, train_calibration as t; t.verified_decision(sys.argv[1], False)' "$1")
}

zm_commit() { cat "$ZIPMIX_CODE/../../../COMMIT" 2>/dev/null || git -C "$ZIPMIX_CODE" rev-parse --short HEAD 2>/dev/null || echo unknown; }
