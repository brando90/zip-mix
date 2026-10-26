#!/usr/bin/env bash
# Run on a fresh 1x A100 cloud instance (Lambda gpu_1x_a100 or similar) after ssh-ing in:
#   curl -fsSL https://raw.githubusercontent.com/brando90/zip-mix/<branch>/experiments/07_training_calibration/expt_v1/launch/lambda_bootstrap.sh | bash -s -- calibration
# or copy this file over and run: bash lambda_bootstrap.sh calibration [git-ref]
# Lambda Stack already ships CUDA PyTorch, so the venv reuses system packages.
# Stage B needs Stage A's decision.json in ~/zipmix_exp07/runs/calibration/ on the same disk
# (or copied back there) before: bash lambda_bootstrap.sh stage_b
# Terminate the instance when analysis.json/decision.json exists; billing is per minute.
set -euo pipefail
stage=${1:?usage: lambda_bootstrap.sh calibration|stage_b [git-ref]}
ref=${2:-calib/exp07-fresh-calibration}
repo=$HOME/zip-mix
if [[ ! -d $repo/.git ]]; then
  git clone --quiet https://github.com/brando90/zip-mix.git "$repo"
fi
git -C "$repo" fetch --quiet origin "$ref" && git -C "$repo" checkout --quiet FETCH_HEAD
export ZIPMIX_WORK=${ZIPMIX_WORK:-$HOME/zipmix_exp07}
exec "$repo/experiments/07_training_calibration/expt_v1/launch/run_stage.sh" "$stage"
