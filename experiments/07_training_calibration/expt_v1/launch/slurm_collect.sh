#!/usr/bin/env bash
# Pack one stage's results (ledger, per-cell result.json and prediction files, decision.json/analysis.json,
# logs, job ids, env_ok.json; no checkpoints or weights) into $ZIPMIX_WORK/exp07_<stage>_results.tgz and print
# the OnDemand link that downloads it (?download=1).
#   slurm_collect.sh calibration | stage_b
set -euo pipefail
stage=${1:?usage: slurm_collect.sh calibration|stage_b}
[[ $stage == calibration || $stage == stage_b ]] || { echo "unknown stage: $stage" >&2; exit 64; }
export ZIPMIX_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ZIPMIX_CODE/launch/slurm_common.sh"
[[ -d $ZIPMIX_WORK/runs/$stage ]] || zm_die "no $ZIPMIX_WORK/runs/$stage"
out=$ZIPMIX_WORK/exp07_${stage}_results.tgz
parts=("runs/$stage" logs env_ok.json)
for f in "$ZIPMIX_WORK"/jobs_*.txt; do [[ -f $f ]] && parts+=("$(basename "$f")"); done
tar czf "$out" -C "$ZIPMIX_WORK" --exclude=checkpoint.pt --exclude=model.pt --exclude='*.tmp' --exclude=.owner.lock \
  --exclude=launch.claim "${parts[@]}"
sha256sum "$out"
ls -l "$out"
for f in decision.json analysis.json; do [[ -f $ZIPMIX_WORK/runs/$stage/$f ]] && echo "has $f"; done
echo "download: $ZM_ONDEMAND/pun/sys/dashboard/files/fs/$out?download=1"
