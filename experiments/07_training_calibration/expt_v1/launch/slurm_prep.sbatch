#!/usr/bin/env bash
# CPU job: build this experiment's own venv, check the shipped data against the committed manifests,
# fetch the pinned Qwen2.5-0.5B weights into HF_HOME, run the CPU tests and write $ZIPMIX_WORK/env_ok.json.
# Submit through slurm_submit.sh (FarmShare: sbatch on -p normal; Sherlock: srun -p dev, which refuses sbatch).
#SBATCH --job-name=zm07-prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
set -euo pipefail
: "${ZIPMIX_CODE:?submit through launch/slurm_submit.sh}"
source "$ZIPMIX_CODE/launch/slurm_common.sh"
echo "prep ${SLURM_JOB_ID:-local} on $(hostname -s) ($ZIPMIX_CLUSTER), code $(zm_commit)"
zm_build_venv
zm_activate
zm_check_data
(cd "$ZIPMIX_CODE" && "$PY" -c '
import sys
from huggingface_hub import snapshot_download
from calib_common import MODEL, MODEL_REVISION
print("model", snapshot_download(MODEL, revision=MODEL_REVISION, cache_dir=sys.argv[1]))
' "$HF_HOME/hub")
(cd "$ZIPMIX_CODE" && "$PY" -m pytest -q -p no:cacheprovider test_calibration.py)
"$PY" - "$ZIPMIX_WORK/env_ok.json" "$(zm_commit)" <<'EOF'
import json, sys, time
import numpy, torch, transformers
json.dump({"code": sys.argv[2], "cluster_job": __import__("os").environ.get("SLURM_JOB_ID"),
           "torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__,
           "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(sys.argv[1], "w"), indent=1)
EOF
echo "PREP_OK $(cat "$ZIPMIX_WORK/env_ok.json" | tr -d '\n ')"
