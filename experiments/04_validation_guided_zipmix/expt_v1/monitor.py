#!/usr/bin/env python3
"""Supervise one owned command without models, retries, or shell interpolation.

Receipts belong in a private runtime directory. Exit zero means the command and
expected cell counts succeeded; it does not establish a scientific conclusion.
A disconnected coordinator does not cancel the command. Reboot recovery is not
provided. A fresh runtime directory is required for every distinct attempt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(262145)
        if len(raw) > 262144:
            return {}, "progress exceeds 256 KiB"
        value = json.loads(raw)
        return (value, None) if isinstance(value, dict) else ({}, "progress is not an object")
    except (OSError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def progress_counts(raw):
    """Accept a compact counter receipt or the training controller's cell manifest."""
    result = {key: raw[key] for key in ("run_id", "fingerprint", "phase", "completed_cells",
              "failed_cells", "total_cells", "expected_cells", "full_matrix_complete",
              "full_clean_matrix", "cells_with_recovery") if key in raw}
    if "expected_training_cells" in raw:
        result["expected_cells"] = raw["expected_training_cells"]
    if "base" in raw:
        result["required_baseline_status"] = (raw["base"].get("status")
                                               if isinstance(raw["base"], dict) else "invalid")
    if "cells" in raw:
        cells = raw["cells"]
        allowed = {"pending", "running", "complete", "failed", "interrupted"}
        if (not isinstance(cells, list) or any(not isinstance(cell, dict)
                or cell.get("status") not in allowed for cell in cells)):
            return result, "invalid cell manifest"
        recovery_counts = [cell.get("recoveries", len(cell.get("resumes", []))) for cell in cells]
        if any(type(count) is not int or count < 0 for count in recovery_counts):
            return result, "invalid recovery counts"
        result.update(completed_cells=sum(cell["status"] == "complete" for cell in cells),
                      failed_cells=sum(cell["status"] in {"failed", "interrupted"} for cell in cells),
                      total_cells=len(cells),
                      recovery_count=sum(recovery_counts),
                      cells_with_recovery=sum(count > 0 for count in recovery_counts))
    return result, None


def identity(pid):
    value = {"pid": pid}
    try:
        value["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        value["start_ticks"] = fields[19]
    except (OSError, IndexError):
        value["identity_unavailable"] = True
    return value


def active_group(pgid):
    """Linux process-group members excluding zombies; never match by command text."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def gpu_receipt(gpu):
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", gpu,
             "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        return {"exit_code": result.returncode, "csv": result.stdout.strip()[:2048],
                "error": result.stderr.strip()[:512]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def terminate_group(process, grace_seconds):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + grace_seconds
    while time.monotonic() < end:
        process.poll()
        if not active_group(process.pid):
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--gpu", required=True, help="One physical GPU index or UUID")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--grace-seconds", type=float, default=30)
    parser.add_argument("--expected-cells", type=int)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--board-session")
    parser.add_argument("--board-dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command or args.max_seconds <= 0 or not 0 < args.poll_seconds <= 120:
        parser.error("command, positive deadline and poll interval in (0,120] required")
    if args.grace_seconds < 0 or (args.expected_cells is not None and args.expected_cells < 1):
        parser.error("grace must be nonnegative and expected cells positive")
    if "," in args.gpu or not args.gpu.strip():
        parser.error("exactly one GPU is required")
    if bool(args.board_session) != bool(args.board_dir):
        parser.error("board-session and board-dir must be provided together")
    return args


def main():
    args = parse_args()
    os.umask(0o077)
    runtime = args.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (runtime / "owner.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Runtime already has an active owner", file=sys.stderr)
        return 73
    if (runtime / "status.json").exists():
        print("Runtime contains an earlier attempt; use a fresh runtime directory", file=sys.stderr)
        return 73
    if args.board_dir:
        args.board_dir = args.board_dir.resolve()
        args.board_dir.mkdir(parents=True, exist_ok=True)
        board_lock = (args.board_dir / "zipmix-board-owner.lock").open("a+")
        try:
            fcntl.flock(board_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Board receipt directory already has an active owner", file=sys.stderr)
            return 73
    run_id = args.run_id or str(uuid.uuid4())
    start = time.monotonic()
    started_epoch = time.time()
    owner = identity(os.getpid())
    process = None
    interrupted = []
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: interrupted.append(number))
    state = {"run_id": run_id, "started_at": timestamp(), "at": timestamp(),
             "phase": "starting", "cwd": str(args.cwd.resolve()), "coordinator": owner,
             "max_seconds": args.max_seconds, "gpu": args.gpu,
             "expected_cells": args.expected_cells, "completed_cells": 0,
             "failed_cells": 0, "reboot_recovery": False,
             "completion_verified": False, "exit_code": None}
    atomic_json(runtime / "command.json", {"run_id": run_id, "argv": args.command})
    events = (runtime / "events.jsonl").open("a", buffering=1)
    output = (runtime / "command.log").open("a", buffering=1)

    def publish(phase, terminal=False):
        state.update(at=timestamp(), phase=phase, elapsed_seconds=time.monotonic()-start)
        raw, progress_error = read_json(args.progress_json) if args.progress_json else ({}, None)
        raw, shape_error = progress_counts(raw)
        progress_error = progress_error or shape_error
        for count_key in ("expected_cells", "total_cells"):
            if (args.expected_cells is not None and count_key in raw
                    and raw[count_key] != args.expected_cells):
                progress_error = f"{count_key} differs from frozen expected count"
        for key in ("completed_cells", "failed_cells"):
            val = raw.get(key)
            if type(val) is int and val >= 0:
                state[key] = max(state[key], val)
        state["progress_error"] = progress_error
        state["raw_progress"] = raw
        state["progress_fresh"] = False
        if args.progress_json:
            try:
                state["progress_fresh"] = args.progress_json.stat().st_mtime >= started_epoch
            except OSError:
                pass
        state["progress_run_binding"] = raw.get("run_id") == run_id
        if process and state.get("driver") is None:
            state["driver"] = identity(process.pid)
        elif not process:
            state["driver"] = None
        state["driver_alive"] = process is not None and process.poll() is None
        state["group_alive_pids"] = active_group(process.pid) if process else []
        state["gpu_receipt"] = gpu_receipt(args.gpu)
        if args.expected_cells is not None:
            state["missing_cells"] = max(0, args.expected_cells-state["completed_cells"]-state["failed_cells"])
        atomic_json(runtime / "status.json", state)
        atomic_json(runtime / "heartbeat.json", {k: state[k] for k in (
            "run_id", "at", "phase", "coordinator", "driver", "driver_alive", "elapsed_seconds")})
        atomic_json(runtime / "monotonic_cells.json", {k: state[k] for k in (
            "run_id", "at", "completed_cells", "failed_cells", "expected_cells")})
        events.write(json.dumps({k: state[k] for k in (
            "run_id", "at", "phase", "driver_alive", "completed_cells", "failed_cells",
            "elapsed_seconds", "gpu_receipt")}) + "\n")
        if args.board_dir:
            base = args.board_dir
            atomic_json(base / "acknowledgement.json", {
                "run_id": run_id, "session": args.board_session, "cwd": state["cwd"],
                "at": state["at"], "phase": phase, "coordinator_pid": owner["pid"],
                "coordinator_start_ticks": owner.get("start_ticks"), "boot_id": owner.get("boot_id")})
            atomic_json(base / "progress.json", {"run_id": run_id, "at": state["at"],
                "phase": phase, "generated": state["completed_cells"],
                "scored": state["completed_cells"], "planned": args.expected_cells,
                "failed": state["failed_cells"], "missing": state.get("missing_cells")})
            atomic_json(base / "driver_identity.json", {
                "run_id": run_id, "at": state["at"], "process": state["driver"]})
            atomic_json(base / "watchdog.json", {"run_id": run_id, "at": state["at"],
                "status": phase, "active": not terminal, "process": owner})

    try:
        publish("starting")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE="1",
                           PYTHONUNBUFFERED="1", ZIPMIX_RUN_ID=run_id)
        process = subprocess.Popen(args.command, cwd=args.cwd, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        publish("running")
        while True:
            if interrupted:
                state["signal"] = interrupted[0]
                terminate_group(process, args.grace_seconds)
                state["exit_code"] = process.poll()
                publish("cancelled", terminal=True)
                return 128+interrupted[0]
            if time.monotonic()-start >= args.max_seconds:
                terminate_group(process, args.grace_seconds)
                state["exit_code"] = process.poll()
                publish("deadline_exceeded", terminal=True)
                return 124
            result = process.poll()
            if result is not None and not active_group(process.pid):
                state["exit_code"] = result
                publish("checking_completion")
                counts_ok = (args.expected_cells is not None
                             and state["completed_cells"] == args.expected_cells
                             and state["progress_error"] is None
                             and state["progress_fresh"]
                             and state["raw_progress"].get("completed_cells") == args.expected_cells
                             and state["raw_progress"].get("failed_cells") == 0
                             and state["raw_progress"].get("required_baseline_status", "complete") == "complete"
                             and ("run_id" not in state["raw_progress"] or state["progress_run_binding"]))
                state["completion_verified"] = result == 0 and counts_ok
                state["clean_completion"] = bool(state["completion_verified"]
                    and state["failed_cells"] == 0
                    and state["raw_progress"].get("recovery_count", 0) == 0
                    and state["raw_progress"].get("full_clean_matrix") is not False)
                success_phase = ("procedure_complete" if state["clean_completion"]
                                 else "procedure_complete_with_recovery")
                phase = (success_phase if state["completion_verified"] else
                         "process_failed" if result else "exited_completion_unverified")
                publish(phase, terminal=True)
                return 0 if state["completion_verified"] else (result or 3)
            publish("running" if result is None else "waiting_for_owned_descendants")
            time.sleep(min(args.poll_seconds, max(0.01, args.max_seconds-(time.monotonic()-start))))
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        terminate_group(process, args.grace_seconds)
        state["exit_code"] = process.poll() if process else None
        try:
            publish("monitor_failed", terminal=True)
        except Exception:
            pass
        raise
    finally:
        events.close()
        output.close()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
