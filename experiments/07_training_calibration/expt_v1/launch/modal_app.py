"""Run one Experiment 07 stage on a single Modal A100 (cloud credits).

From the repository root, with `modal token new` done once:

    modal run experiments/07_training_calibration/expt_v1/launch/modal_app.py --stage calibration
    modal run experiments/07_training_calibration/expt_v1/launch/modal_app.py --stage stage_b
    modal volume get zipmix-exp07 runs/calibration/decision.json .

Data, runs and the model cache live on the `zipmix-exp07` volume, never in the
repository. Stage B reads Stage A's decision.json from the same volume. A
re-run resumes the ledger (each interrupted cell resumes at most once).
"""
from pathlib import Path
import subprocess

import modal

EXPERIMENTS = Path(__file__).resolve().parents[3]
CODE = "/root/experiments"
WORK = "/vol"
HOME = f"{CODE}/07_training_calibration/expt_v1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2,<3", "transformers>=4.45,<5", "datasets>=2.20,<5", "huggingface_hub>=0.24,<2",
                 "numpy>=1.24,<3", "scipy>=1.10,<2", "lz4>=4,<5")
    .env({"HF_HOME": f"{WORK}/hf"})
    # Code only: prepared data, runs, caches and weights are excluded.
    .add_local_dir(EXPERIMENTS, CODE, ignore=["**/data", "**/runs", "**/cache", "**/__pycache__",
                                               "**/*.pt", "**/*.safetensors", "**/*.png", "**/*.pdf"])
)
volume = modal.Volume.from_name("zipmix-exp07", create_if_missing=True)
app = modal.App("zipmix-exp07", image=image)


def sh(*args):
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=HOME).returncode


@app.function(gpu="A100-40GB", volumes={WORK: volume}, timeout=4 * 3600)
def run_stage(stage: str) -> int:
    if stage not in ("calibration", "stage_b"):
        raise ValueError(stage)
    if not Path(f"{WORK}/data/stage_b/manifest.json").exists():
        if sh("python", "prepare_calibration.py", "--output", f"{WORK}/data"):
            raise RuntimeError("preparation failed; no training started")
        volume.commit()
    extra = []
    if stage == "stage_b":
        decision = f"{WORK}/runs/calibration/decision.json"
        if not Path(decision).exists():
            raise RuntimeError("Stage A decision.json missing on the volume; run --stage calibration first")
        extra = ["--decision", decision]
    run = f"{WORK}/runs/{stage}"
    status = sh("python", "train_calibration.py", "--data", f"{WORK}/data/{stage}", "--output", run,
                "--cache-dir", f"{WORK}/hf", *extra)
    sh("python", "analyze_calibration.py", "--run", run, "--stage", stage)
    volume.commit()
    return status


@app.local_entrypoint()
def main(stage: str = "calibration"):
    status = run_stage.remote(stage)
    result = "decision.json" if stage == "calibration" else "analysis.json"
    print(f"{stage} exit {status}; fetch with: modal volume get zipmix-exp07 runs/{stage}/{result} .")
