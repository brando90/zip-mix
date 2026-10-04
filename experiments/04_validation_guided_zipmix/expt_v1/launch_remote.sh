#!/usr/bin/env bash
# Run this on the verified compute host after syncing immutable inputs.
# Usage: PYTHON_BIN=/path/to/venv/bin/python ./launch_remote.sh PRIVATE_RUN_DIR \
#   --max-seconds 43200 --gpu 0 --expected-cells 27 --progress-json /path/counts.json \
#   --cwd /path/to/repo -- /path/to/venv/bin/python /path/to/train.py [arguments]
# To populate an existing board-visible tmux session, additionally pass
# --board-session SESSION --board-dir REPO_PARENT; the session pane must use REPO.
# This survives logout; it does not install reboot recovery or restart failed cells.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: launch_remote.sh PRIVATE_RUN_DIR MONITOR_ARGUMENTS -- COMMAND [ARGUMENTS]' >&2
  exit 64
fi
runtime_dir=$1
shift
python_bin=${PYTHON_BIN:-python3}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
umask 077
mkdir -p -- "$runtime_dir"
runtime_dir=$(cd -- "$runtime_dir" && pwd)
# mkdir is atomic; reject duplicate launch attempts without replacing any receipt.
if ! mkdir -- "$runtime_dir/launch.claim"; then
  echo 'Launch already claimed; inspect its receipt or use a fresh runtime directory' >&2
  exit 73
fi
nohup "$python_bin" "$script_dir/monitor.py" --runtime-dir "$runtime_dir" "$@" \
  </dev/null >"$runtime_dir/supervisor.log" 2>&1 &
supervisor_pid=$!
printf '%s\n' "$supervisor_pid" >"$runtime_dir/supervisor.pid"
# This verifies only startup, not training completion or scientific success.
for ((attempt=0; attempt<50; attempt++)); do
  if [[ -s "$runtime_dir/heartbeat.json" ]]; then
    printf 'Supervisor started: pid=%s receipt=%s\n' "$supervisor_pid" "$runtime_dir/heartbeat.json"
    exit 0
  fi
  if ! kill -0 "$supervisor_pid" 2>/dev/null; then
    echo "Supervisor exited during startup; inspect $runtime_dir/supervisor.log" >&2
    exit 1
  fi
  sleep 0.2
done
printf 'Supervisor process exists but startup receipt is unverified: pid=%s log=%s\n' \
  "$supervisor_pid" "$runtime_dir/supervisor.log" >&2
exit 1
