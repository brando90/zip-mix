#!/usr/bin/env bash
# Copy the diagnostic run's receipts (no checkpoints or weights) next to this report and analyze them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${1:-$HERE/../expt_v1/runs_diag/diag_lr5e-6_10-21-2026}
OUT=$HERE/diagnostic_lr5e-6
mkdir -p "$OUT"
cp "$RUN"/ledger.json "$RUN"/train.log "$RUN"/launch.sh "$RUN"/exit-code "$OUT"/
for cell in "$RUN"/base "$RUN"/*__seed*; do
  name=$(basename "$cell"); mkdir -p "$OUT/$name"
  cp "$cell"/result.json "$cell"/*_predictions.npz "$OUT/$name"/
done
PY=${PYTHON:-python3}
"$PY" "$HERE/analyze_diagnostic.py" --run "$RUN" --out "$OUT"
( cd "$OUT" && find . -type f | sort | xargs sha256sum > SHA256SUMS )
echo "collected into $OUT"
