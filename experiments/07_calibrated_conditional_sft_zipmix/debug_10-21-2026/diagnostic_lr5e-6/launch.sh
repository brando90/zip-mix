#!/usr/bin/env bash
# Pre-registered Exp 07 diagnostic (10-21-2026): 3 selectors x 5 seeds at lr 5e-6, eval on cal_target + cal_retention only.
cd /lfs/skampere2/0/brando9/snap_jobs/codex-zipmix-e07-debug-10-21-2026/repo/experiments/07_calibrated_conditional_sft_zipmix/expt_v1
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH
date -u +"start %FT%TZ" >> runs_diag/diag_lr5e-6_10-21-2026/train.log
timeout --signal=TERM 1800 /lfs/skampere2/0/brando9/miniconda/envs/self_opt_data_gen/bin/python train_exp07.py --mode diagnostic --learning-rate 5e-6 \
  --methods conditional_aligned stratum_uniform conditional_shuffled --seeds 0 1 2 3 4 \
  --eval-splits cal_target cal_retention --data data --output runs_diag/diag_lr5e-6_10-21-2026 \
  --cache-dir /lfs/skampere2/0/brando9/.cache/huggingface/hub >> runs_diag/diag_lr5e-6_10-21-2026/train.log 2>&1
rc=$?
date -u +"end %FT%TZ exit=$rc" >> runs_diag/diag_lr5e-6_10-21-2026/train.log
echo $rc > runs_diag/diag_lr5e-6_10-21-2026/exit-code
